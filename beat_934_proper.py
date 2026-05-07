"""
Proper ensemble to beat 0.934 using sns53 (0.934) as pseudo-labels.

Key strategy:
1. sns53 gives us 93.4%-accurate pseudo-labels for all 2000 test points
2. Augment training with these pseudo-labels
3. Use geographic LOO-KNN on augmented dataset to flag spatial inconsistencies
4. Only flip sns53 predictions when spatial evidence is extremely strong
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

np.random.seed(42)
BASE = '/home/user/EY-Biodiversity-/'

# ── Load data ────────────────────────────────────────────────────────────────
train   = pd.read_csv(BASE + 'Training_Data.csv')
test    = pd.read_csv(BASE + 'Test.csv')
sns53   = pd.read_csv(BASE + 'submission_sns53.csv')

train_coords = train[['Latitude', 'Longitude']].values
test_coords  = test[['Latitude', 'Longitude']].values
y_train      = train['Occurrence Status'].values

# Align sns53 to test order
id_to_sns53 = dict(zip(sns53['ID'], sns53['Target']))
sns53_labels = np.array([id_to_sns53[id_] for id_ in test['ID']])

print(f"Train: {len(train)} (pos={y_train.mean():.3f})")
print(f"Test: {len(test)}")
print(f"sns53 pos rate: {sns53_labels.mean():.3f} ({sns53_labels.sum()} positive)")

pos_coords = train_coords[y_train == 1]
neg_coords = train_coords[y_train == 0]
lat_m, lat_s = train_coords[:, 0].mean(), train_coords[:, 0].std()
lon_m, lon_s = train_coords[:, 1].mean(), train_coords[:, 1].std()

def normalize(coords):
    return np.column_stack([(coords[:,0]-lat_m)/lat_s, (coords[:,1]-lon_m)/lon_s])

train_norm = normalize(train_coords)
test_norm  = normalize(test_coords)

# ── Build augmented dataset (training + sns53 pseudo-labels) ─────────────────
aug_coords = np.vstack([train_coords, test_coords])
aug_labels = np.concatenate([y_train, sns53_labels])
aug_norm   = normalize(aug_coords)
is_train   = np.concatenate([np.ones(len(train)), np.zeros(len(test))]).astype(bool)
is_test    = ~is_train

n_train = len(train)
n_test  = len(test)
n_total = len(aug_coords)

print(f"\nAugmented dataset: {n_total} points")

# ── Approach 1: LOO-KNN on augmented dataset ─────────────────────────────────
# For each TEST point i: find k nearest neighbors EXCLUDING point i itself
# Using distance-weighted KNN with training points weighted higher (trust=1.0)
# vs pseudo-labeled test points (trust = 0.934 on average)
print("\n=== Building spatial augmented LOO predictions ===")

TRUST_TRAIN = 1.0
TRUST_TEST  = 0.934   # approximate accuracy of sns53 pseudo-labels

k_values = [5, 10, 15, 20, 30]
aug_knn_preds = {}

for k in k_values:
    test_prob_k = np.zeros(n_test)
    for i in range(n_test):
        # Index in augmented array
        aug_idx = n_train + i
        # Build candidate set (all except test point i)
        mask = np.arange(n_total) != aug_idx
        cand_norm   = aug_norm[mask]
        cand_labels = aug_labels[mask]
        cand_is_tr  = is_train[mask]

        # Find k nearest
        tree = cKDTree(cand_norm)
        dists, idxs = tree.query(aug_norm[aug_idx], k=k)
        
        # Distance-weighted votes, with trust weighting
        eps = 1e-9
        weights = 1.0 / (dists + eps)
        trust = np.where(cand_is_tr[idxs], TRUST_TRAIN, TRUST_TEST)
        w_total = weights * trust
        
        vote_pos = (w_total * (cand_labels[idxs] == 1)).sum()
        vote_neg = (w_total * (cand_labels[idxs] == 0)).sum()
        test_prob_k[i] = vote_pos / (vote_pos + vote_neg + eps)
    
    aug_knn_preds[k] = test_prob_k
    flip_to1 = ((sns53_labels==0) & (test_prob_k>0.80)).sum()
    flip_to0 = ((sns53_labels==1) & (test_prob_k<0.20)).sum()
    print(f"  k={k:3d}: would flip to 1: {flip_to1}, flip to 0: {flip_to0}, "
          f"avg prob where sns53=0: {test_prob_k[sns53_labels==0].mean():.3f}, "
          f"avg prob where sns53=1: {test_prob_k[sns53_labels==1].mean():.3f}")

# ── Approach 2: LightGBM on augmented geo features ───────────────────────────
def build_geo_features(coords):
    lat = coords[:, 0]; lon = coords[:, 1]
    ln  = (lat - lat_m) / lat_s; lo = (lon - lon_m) / lon_s
    d_pc = np.sqrt((lat - pos_coords[:,0].mean())**2 + (lon - pos_coords[:,1].mean())**2)
    d_nc = np.sqrt((lat - neg_coords[:,0].mean())**2 + (lon - neg_coords[:,1].mean())**2)
    return pd.DataFrame({
        'lat': lat, 'lon': lon, 'ln': ln, 'lo': lo,
        'ln2': ln**2, 'lo2': lo**2, 'ln3': ln**3, 'lo3': lo**3,
        'ln4': ln**4, 'lo4': lo**4,
        'ln_lo': ln*lo, 'ln2_lo': ln**2*lo, 'ln_lo2': ln*lo**2,
        'ln2_lo2': ln**2*lo**2,
        'dist': np.sqrt(ln**2+lo**2), 'angle': np.arctan2(lo, ln),
        'sin_lat': np.sin(np.radians(lat)), 'cos_lat': np.cos(np.radians(lat)),
        'sin_lon': np.sin(np.radians(lon)), 'cos_lon': np.cos(np.radians(lon)),
        'd_pos_centroid': d_pc, 'd_neg_centroid': d_nc,
        'd_ratio': d_pc / (d_nc + 1e-9),
    })

X_aug_geo  = build_geo_features(aug_coords)
X_test_geo = build_geo_features(test_coords)

# Train LGB on augmented data using sns53 pseudo-labels
# Weight: training points get weight 1.0 / (1 - 0.934) / 0.934 ≈ 1.0 / 0.93 ≈ higher
# Actually just use 1.0 for train and 0.934 for pseudo
sample_weights = np.where(is_train, 1.0, 0.934)

print("\n=== Training LGB on augmented dataset ===")
lgb_params = dict(
    objective='binary', metric='binary_logloss',
    n_estimators=800, learning_rate=0.02,
    num_leaves=63, min_child_samples=15,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.3, reg_lambda=0.3,
    verbose=-1, random_state=42,
)

# 5-fold CV on TRUE training data only to calibrate
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_lgb_aug = np.zeros(n_total)

for fold, (tr_idx, val_idx) in enumerate(skf.split(aug_coords, aug_labels)):
    print(f"  Fold {fold+1}/5...", end=' ', flush=True)
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(X_aug_geo.iloc[tr_idx], aug_labels[tr_idx],
          sample_weight=sample_weights[tr_idx],
          eval_set=[(X_aug_geo.iloc[val_idx], aug_labels[val_idx])],
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)])
    oof_lgb_aug[val_idx] = m.predict_proba(X_aug_geo.iloc[val_idx])[:, 1]
    print(f"val acc={accuracy_score(aug_labels[val_idx], (oof_lgb_aug[val_idx]>0.5).astype(int)):.4f}")

print(f"OOF train acc: {accuracy_score(y_train, (oof_lgb_aug[:n_train]>0.5).astype(int)):.4f}")
print(f"OOF test  acc: {accuracy_score(sns53_labels, (oof_lgb_aug[n_train:]>0.5).astype(int)):.4f}")
lgb_aug_test_prob = oof_lgb_aug[n_train:]

# ── Analyze disagreements ────────────────────────────────────────────────────
print("\n=== Disagreement analysis ===")
knn30_prob = aug_knn_preds[30]

# Points where sns53=1 but aug spatial says ~0
strong_flip_0 = (sns53_labels == 1) & (knn30_prob < 0.20) & (lgb_aug_test_prob < 0.30)
strong_flip_1 = (sns53_labels == 0) & (knn30_prob > 0.80) & (lgb_aug_test_prob > 0.70)
print(f"Strong flip 1→0: {strong_flip_0.sum()} points (knn30<0.20 AND lgb<0.30)")
print(f"Strong flip 0→1: {strong_flip_1.sum()} points (knn30>0.80 AND lgb>0.70)")

# Conservative: only flip when VERY sure
v_strong_flip_0 = (sns53_labels == 1) & (knn30_prob < 0.10) & (lgb_aug_test_prob < 0.20)
v_strong_flip_1 = (sns53_labels == 0) & (knn30_prob > 0.90) & (lgb_aug_test_prob > 0.80)
print(f"Very strong flip 1→0: {v_strong_flip_0.sum()} points")
print(f"Very strong flip 0→1: {v_strong_flip_1.sum()} points")

# ── Build multiple candidate submissions ─────────────────────────────────────
print("\n=== Building candidate submissions ===")

# A: sns53 unchanged (baseline)
pred_A = sns53_labels.copy()
print(f"A (sns53 baseline): {pred_A.sum()} pos, {(pred_A==0).sum()} neg")

# B: Strong flips (k=30 + LGB)
pred_B = sns53_labels.copy()
pred_B[strong_flip_0] = 0
pred_B[strong_flip_1] = 1
print(f"B (strong flip): {pred_B.sum()} pos, {(pred_B==0).sum()} neg, "
      f"{(pred_B!=sns53_labels).sum()} flips")

# C: Very strong flips only
pred_C = sns53_labels.copy()
pred_C[v_strong_flip_0] = 0
pred_C[v_strong_flip_1] = 1
print(f"C (very strong flip): {pred_C.sum()} pos, {(pred_C==0).sum()} neg, "
      f"{(pred_C!=sns53_labels).sum()} flips")

# D: Soft blend (sns53 hard 0.9 + knn30 0.05 + lgb 0.05)
# Treat sns53 hard as probability 0.93/0.07
soft_sns53 = np.where(sns53_labels==1, 0.93, 0.07)
soft_blend = 0.85*soft_sns53 + 0.10*knn30_prob + 0.05*lgb_aug_test_prob
pred_D = (soft_blend > 0.5).astype(int)
print(f"D (soft blend 85/10/5): {pred_D.sum()} pos, flips={( pred_D!=sns53_labels).sum()}")

# E: Multiple thresholds for strong flip
best_sim = 0
best_pred_E = sns53_labels.copy()
best_params_E = None

for knn_flip_thresh in [0.10, 0.15, 0.20, 0.25, 0.30]:
    for lgb_flip_thresh in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        for knn_k in [15, 20, 30]:
            knn_prob_k = aug_knn_preds[knn_k]
            f0 = (sns53_labels==1) & (knn_prob_k < knn_flip_thresh) & (lgb_aug_test_prob < lgb_flip_thresh)
            f1 = (sns53_labels==0) & (knn_prob_k > (1-knn_flip_thresh)) & (lgb_aug_test_prob > (1-lgb_flip_thresh))
            total_flips = f0.sum() + f1.sum()
            if total_flips == 0:
                continue
            
            # Score proxy: confident flips relative to total flips
            # We want flips that are in high-agreement zones
            conf_f0 = (knn_prob_k[f0]).mean() if f0.sum() > 0 else 0.5
            conf_f1 = (1 - knn_prob_k[f1]).mean() if f1.sum() > 0 else 0.5
            avg_flip_conf = (conf_f0 * f0.sum() + conf_f1 * f1.sum()) / (total_flips + 1e-9)
            
            # Avoid flipping too many (risky)
            flip_penalty = max(0, total_flips - 30) * 0.01
            score = (1 - avg_flip_conf) - flip_penalty  # lower avg prob = more confident
            
            if score > best_sim:
                best_sim = score
                pred_E = sns53_labels.copy()
                pred_E[f0] = 0
                pred_E[f1] = 1
                best_pred_E = pred_E.copy()
                best_params_E = (knn_k, knn_flip_thresh, lgb_flip_thresh, total_flips)

k_e, t_knn_e, t_lgb_e, flips_e = best_params_E
print(f"E (optimized flip): k={k_e}, knn_thresh={t_knn_e}, lgb_thresh={t_lgb_e}, "
      f"flips={flips_e}, pos={best_pred_E.sum()}")

# ── CV validation: simulate the approach on training data ─────────────────────
print("\n=== CV validation on training data ===")
# Simulate: 
# - Take training labels as ground truth
# - Introduce 6.6% noise (like sns53 quality) as "pseudo-labels"  
# - Apply LOO-KNN augmented correction
# - Measure improvement over noisy labels

n_sim = 3
sim_results = {'noisy': [], 'corrected': []}

for sim in range(n_sim):
    rng = np.random.RandomState(sim * 123 + 7)
    
    # Create "noisy" labels (simulates sns53 quality)
    noisy = y_train.copy()
    flip_idx = rng.choice(len(y_train), size=int(0.066*len(y_train)), replace=False)
    noisy[flip_idx] = 1 - noisy[flip_idx]
    
    # Apply spatial correction (leave-one-out on training data only)
    # Use k=20 with training data
    k_cv = 20
    corrected = noisy.copy()
    tree = cKDTree(train_norm)
    
    for i in range(len(train)):
        # k+1 because first neighbor is self
        dists, idxs = tree.query(train_norm[i], k=k_cv+1)
        # exclude self
        mask = idxs != i
        idxs = idxs[mask][:k_cv]
        dists = dists[mask][:k_cv]
        
        eps = 1e-9
        weights = 1.0 / (dists + eps)
        pos_w = (weights[noisy[idxs]==1]).sum()
        neg_w = (weights[noisy[idxs]==0]).sum()
        prob = pos_w / (pos_w + neg_w + eps)
        
        # Only flip if very confident
        if noisy[i] == 1 and prob < 0.15:
            corrected[i] = 0
        elif noisy[i] == 0 and prob > 0.85:
            corrected[i] = 1
    
    noisy_acc = accuracy_score(y_train, noisy)
    corr_acc  = accuracy_score(y_train, corrected)
    
    n_flipped = (corrected != noisy).sum()
    n_correct_flips = ((corrected != noisy) & (corrected == y_train)).sum()
    
    sim_results['noisy'].append(noisy_acc)
    sim_results['corrected'].append(corr_acc)
    print(f"  Sim {sim+1}: noisy={noisy_acc:.4f} -> corrected={corr_acc:.4f} "
          f"(flips={n_flipped}, correct_flips={n_correct_flips})")

print(f"\nMean noisy acc: {np.mean(sim_results['noisy']):.4f}")
print(f"Mean corrected acc: {np.mean(sim_results['corrected']):.4f}")
print(f"Mean improvement: {np.mean(sim_results['corrected']) - np.mean(sim_results['noisy']):.5f}")

# ── Final selection ────────────────────────────────────────────────────────────
print("\n=== FINAL APPROACH SUMMARY ===")
for name, preds in [('A sns53', pred_A), ('B strong', pred_B),
                    ('C very strong', pred_C), ('D soft', pred_D), ('E optimized', best_pred_E)]:
    flips = (preds != sns53_labels).sum()
    print(f"  {name:<18}: pos={preds.sum():4d}, flips={flips:3d}")

# Save approach C (very conservative) as primary bet934 submission
# and approach B as secondary
submission_C = pd.DataFrame({'ID': test['ID'], 'Target': pred_C})
submission_C.to_csv(BASE + 'submission_beat934.csv', index=False)
print(f"\nSaved submission_beat934.csv (very conservative approach C):")
print(f"  {pred_C.sum()} positive, {(pred_C==0).sum()} negative")
print(f"  {(pred_C!=sns53_labels).sum()} flips from sns53")

submission_B = pd.DataFrame({'ID': test['ID'], 'Target': pred_B})
submission_B.to_csv(BASE + 'submission_beat934_B.csv', index=False)
print(f"\nSaved submission_beat934_B.csv (strong approach B):")
print(f"  {pred_B.sum()} positive, {(pred_B==0).sum()} negative")
print(f"  {(pred_B!=sns53_labels).sum()} flips from sns53")

submission_E = pd.DataFrame({'ID': test['ID'], 'Target': best_pred_E})
submission_E.to_csv(BASE + 'submission_beat934_E.csv', index=False)
print(f"\nSaved submission_beat934_E.csv (optimized approach E):")
print(f"  {best_pred_E.sum()} positive, {(best_pred_E==0).sum()} negative")
print(f"  {(best_pred_E!=sns53_labels).sum()} flips from sns53")

print("\n=== DONE ===")
