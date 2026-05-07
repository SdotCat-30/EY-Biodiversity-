"""
Ensemble script to beat 0.934 on EY Biodiversity challenge.

Strategy:
1. Re-build spatial model with full probability outputs
2. Use Sanskar (sns53-equivalent) submission as strong pseudo-label anchor (0.934)
3. Soft blend: sns53 predictions (as probabilities) + spatial model probs
4. Semi-supervised: augment training with high-confidence test pseudo-labels
5. Confidence-based flipping of sns53 on points where spatial model is very confident
6. CV simulation with noisy labels (~6.6% noise) to pick best approach
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

np.random.seed(42)

# ─── Load data ──────────────────────────────────────────────────────────────
BASE = '/home/user/EY-Biodiversity-/'

train = pd.read_csv(BASE + 'Training_Data.csv')
test  = pd.read_csv(BASE + 'Test.csv')
improved = pd.read_csv(BASE + 'submission_improved.csv')
sanskar  = pd.read_csv(BASE + 'submission_sanskar.csv')

train_coords = train[['Latitude', 'Longitude']].values
test_coords  = test[['Latitude', 'Longitude']].values
y_train      = train['Occurrence Status'].values

pos_coords = train_coords[y_train == 1]
neg_coords = train_coords[y_train == 0]

print(f"Train: {len(train)} | Test: {len(test)}")
print(f"Train pos rate: {y_train.mean():.4f}")
print(f"Improved pos rate: {improved.Target.mean():.4f}")
print(f"Sanskar  pos rate: {sanskar.Target.mean():.4f}")

# ─── Coordinate normalisation ────────────────────────────────────────────────
lat_m, lat_s = train_coords[:, 0].mean(), train_coords[:, 0].std()
lon_m, lon_s = train_coords[:, 1].mean(), train_coords[:, 1].std()

def normalize(coords):
    return np.column_stack([
        (coords[:, 0] - lat_m) / lat_s,
        (coords[:, 1] - lon_m) / lon_s,
    ])

train_norm = normalize(train_coords)
test_norm  = normalize(test_coords)

# ─── Rich geo features ───────────────────────────────────────────────────────
def build_geo_features(coords):
    lat = coords[:, 0]
    lon = coords[:, 1]
    ln  = (lat - lat_m) / lat_s
    lo  = (lon - lon_m) / lon_s
    d_pc = np.sqrt((lat - pos_coords[:, 0].mean())**2 + (lon - pos_coords[:, 1].mean())**2)
    d_nc = np.sqrt((lat - neg_coords[:, 0].mean())**2 + (lon - neg_coords[:, 1].mean())**2)
    d_ne = np.sqrt((lat - (-30.92))**2 + (lon - 151.48)**2)
    d_sw = np.sqrt((lat - (-39.74))**2 + (lon - 139.94)**2)
    d_nw = np.sqrt((lat - (-30.92))**2 + (lon - 139.94)**2)
    d_se = np.sqrt((lat - (-39.74))**2 + (lon - 151.48)**2)
    return pd.DataFrame({
        'lat': lat, 'lon': lon,
        'ln': ln, 'lo': lo,
        'ln2': ln**2, 'lo2': lo**2,
        'ln3': ln**3, 'lo3': lo**3,
        'ln4': ln**4, 'lo4': lo**4,
        'ln5': ln**5, 'lo5': lo**5,
        'ln_lo':   ln*lo,
        'ln2_lo':  ln**2*lo,   'ln_lo2':  ln*lo**2,
        'ln3_lo':  ln**3*lo,   'ln_lo3':  ln*lo**3,
        'ln2_lo2': ln**2*lo**2,
        'dist':  np.sqrt(ln**2 + lo**2),
        'angle': np.arctan2(lo, ln),
        'sin_lat': np.sin(np.radians(lat)), 'cos_lat': np.cos(np.radians(lat)),
        'sin_lon': np.sin(np.radians(lon)), 'cos_lon': np.cos(np.radians(lon)),
        'sin2_lat': np.sin(2*np.radians(lat)), 'cos2_lat': np.cos(2*np.radians(lat)),
        'sin2_lon': np.sin(2*np.radians(lon)), 'cos2_lon': np.cos(2*np.radians(lon)),
        'd_pos_centroid': d_pc, 'd_neg_centroid': d_nc,
        'd_pos_neg_ratio': d_pc / (d_nc + 1e-9),
        'd_ne': d_ne, 'd_sw': d_sw, 'd_nw': d_nw, 'd_se': d_se,
    })

X_train_geo = build_geo_features(train_coords)
X_test_geo  = build_geo_features(test_coords)

# ─── KDE helper ──────────────────────────────────────────────────────────────
def kde_predict(pos_tr, neg_tr, coords_val):
    np_ = len(pos_tr)
    nn_ = len(neg_tr)
    kp  = gaussian_kde(pos_tr.T, bw_method='silverman')
    kn  = gaussian_kde(neg_tr.T, bw_method='silverman')
    pv  = kp(coords_val.T) * np_
    nv  = kn(coords_val.T) * nn_
    return pv / (pv + nv + 1e-12)

# ─── Build spatial model OOF + test probabilities ────────────────────────────
print("\n=== Building spatial model OOF & test probabilities ===")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lgb_params = dict(
    objective='binary', metric='binary_logloss',
    n_estimators=600, learning_rate=0.025,
    num_leaves=63, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=0.5,
    verbose=-1, random_state=42,
)

oof_knn = np.zeros(len(train_coords))
oof_svm = np.zeros(len(train_coords))
oof_lgb = np.zeros(len(train_coords))
oof_kde = np.zeros(len(train_coords))

test_knn = np.zeros(len(test_coords))
test_svm = np.zeros(len(test_coords))
test_lgb = np.zeros(len(test_coords))
test_kde = np.zeros(len(test_coords))

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_coords, y_train)):
    print(f"  Fold {fold+1}/5 ...", flush=True)
    Xtr_n, Xval_n = train_norm[tr_idx], train_norm[val_idx]
    Xtr_g, Xval_g = X_train_geo.iloc[tr_idx], X_train_geo.iloc[val_idx]
    ytr, yval     = y_train[tr_idx], y_train[val_idx]

    # KNN k=7
    knn = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
    knn.fit(Xtr_n, ytr)
    oof_knn[val_idx]  = knn.predict_proba(Xval_n)[:, 1]
    test_knn          += knn.predict_proba(test_norm)[:, 1] / 5

    # SVM RBF
    svm = SVC(C=5.0, gamma=1.0, kernel='rbf', probability=True)
    svm.fit(Xtr_n, ytr)
    oof_svm[val_idx]  = svm.predict_proba(Xval_n)[:, 1]
    test_svm          += svm.predict_proba(test_norm)[:, 1] / 5

    # LGB
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(Xtr_g, ytr,
          eval_set=[(Xval_g, yval)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    oof_lgb[val_idx]  = m.predict_proba(Xval_g)[:, 1]
    test_lgb          += m.predict_proba(X_test_geo)[:, 1] / 5

    # KDE
    pos_tr_c = train_coords[tr_idx][ytr == 1]
    neg_tr_c = train_coords[tr_idx][ytr == 0]
    oof_kde[val_idx]  = kde_predict(pos_tr_c, neg_tr_c, train_coords[val_idx])
    test_kde          += kde_predict(pos_tr_c, neg_tr_c, test_coords) / 5

print("\nOOF accuracies:")
for name, arr in [('KNN', oof_knn), ('SVM', oof_svm), ('LGB', oof_lgb), ('KDE', oof_kde)]:
    acc = accuracy_score(y_train, (arr > 0.5).astype(int))
    print(f"  {name}: {acc:.4f}")

# ─── Spatial ensemble probability ───────────────────────────────────────────
# Weights: LGB=0.6, KNN=0.2, KDE=0.1, SVM=0.1
W_spatial = np.array([0.6, 0.2, 0.1, 0.1])

oof_spatial  = W_spatial[0]*oof_lgb  + W_spatial[1]*oof_knn  + W_spatial[2]*oof_kde  + W_spatial[3]*oof_svm
test_spatial = W_spatial[0]*test_lgb + W_spatial[1]*test_knn + W_spatial[2]*test_kde + W_spatial[3]*test_svm

spatial_oof_acc = accuracy_score(y_train, (oof_spatial > 0.55).astype(int))
print(f"\nSpatial ensemble OOF accuracy (threshold=0.55): {spatial_oof_acc:.4f}")

# ─── sns53 as soft probabilities ─────────────────────────────────────────────
# Treat hard labels as P(y=1): 0.934 accuracy means ~93% of labels correct
# We model the "soft" version: 1 → 0.93, 0 → 0.07  (calibrated)
sns53_hard = sanskar.sort_values('ID').reset_index(drop=True)
test_sorted = test.sort_values('ID').reset_index(drop=True)  # ensure same order
# Re-align submissions to test order
id_to_sns53 = dict(zip(sanskar.ID, sanskar.Target))
id_to_improved = dict(zip(improved.ID, improved.Target))

sns53_labels  = np.array([id_to_sns53[id_]    for id_ in test['ID']])
spatial_labels = (test_spatial > 0.55).astype(int)

# Calibrate sns53 as soft: accuracy ~0.934 so we pull extremes slightly in
SNS53_CONF = 0.93
sns53_soft = np.where(sns53_labels == 1, SNS53_CONF, 1 - SNS53_CONF)

print(f"\nsns53 positive rate: {sns53_labels.mean():.4f}")
print(f"Spatial positive rate: {spatial_labels.mean():.4f}")

# ─── Disagreement analysis ───────────────────────────────────────────────────
disagree_mask = sns53_labels != spatial_labels
print(f"\nSpatial vs sns53 disagreements: {disagree_mask.sum()} ({disagree_mask.mean()*100:.1f}%)")
sns1_sp0 = ((sns53_labels == 1) & (spatial_labels == 0)).sum()
sns0_sp1 = ((sns53_labels == 0) & (spatial_labels == 1)).sum()
print(f"  sns53=1, spatial=0: {sns1_sp0}")
print(f"  sns53=0, spatial=1: {sns0_sp1}")

# ─── Approach A: Majority vote (improved + sanskar + spatial) ────────────────
print("\n\n=== APPROACH A: Majority Vote ===")
improved_labels = np.array([id_to_improved[id_] for id_ in test['ID']])
votes = improved_labels + sns53_labels + spatial_labels
majority = (votes >= 2).astype(int)
print(f"Majority vote positive rate: {majority.mean():.4f}")
print(f"vs sns53: {(majority != sns53_labels).sum()} flips")

# ─── Approach B: Weighted soft voting ────────────────────────────────────────
print("\n=== APPROACH B: Weighted Soft Voting ===")
# sns53=0.934, improved=0.88161209, spatial=0.7849 (OOF proxy)
# Weights proportional to (accuracy - baseline)^2  where baseline=0.6
def weight_from_acc(acc, baseline=0.60):
    return max(0, acc - baseline)**2

w_sns53    = weight_from_acc(0.934)
w_improved = weight_from_acc(0.88161209)
w_spatial  = weight_from_acc(0.7849)
total_w    = w_sns53 + w_improved + w_spatial

w_sns53    /= total_w
w_improved /= total_w
w_spatial  /= total_w

print(f"Weights: sns53={w_sns53:.3f}, improved={w_improved:.3f}, spatial={w_spatial:.3f}")

# Soft: improved hard->soft calibrated
IMP_CONF = 0.88
improved_soft = np.where(improved_labels == 1, IMP_CONF, 1 - IMP_CONF)

soft_blend = w_sns53 * sns53_soft + w_improved * improved_soft + w_spatial * test_spatial
soft_pred_B = (soft_blend > 0.5).astype(int)
print(f"Approach B positive rate: {soft_pred_B.mean():.4f}")
print(f"vs sns53: {(soft_pred_B != sns53_labels).sum()} flips")

# ─── Approach C: Semi-supervised spatial correction ──────────────────────────
print("\n=== APPROACH C: Semi-supervised spatial correction ===")
# Use sns53 as pseudo-labels for test data; add high-confidence ones to training
# Then re-train LGB on augmented set and use to refine predictions

# Pick high-confidence pseudo-labels: spatial model also agrees with sns53
conf_mask = sns53_soft > 0.80  # high confidence = sns53 very sure
agree_mask = sns53_labels == (test_spatial > 0.55).astype(int)  # also spatial agrees
pseudo_mask = conf_mask & agree_mask

print(f"High-confidence pseudo-labels: {pseudo_mask.sum()} test points")

pseudo_coords  = test_coords[pseudo_mask]
pseudo_labels  = sns53_labels[pseudo_mask]

# Augmented training set
aug_coords = np.vstack([train_coords, pseudo_coords])
aug_labels = np.concatenate([y_train, pseudo_labels])
aug_geo    = build_geo_features(aug_coords)

print(f"Augmented train size: {len(aug_coords)} (orig={len(train_coords)}, pseudo={pseudo_mask.sum()})")

# Train LGB on augmented data
test_lgb_aug = np.zeros(len(test_coords))
skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
for fold, (tr_idx, val_idx) in enumerate(skf2.split(aug_coords, aug_labels)):
    Xtr_g = aug_geo.iloc[tr_idx]
    Xval_g = aug_geo.iloc[val_idx]
    ytr = aug_labels[tr_idx]
    yval = aug_labels[val_idx]
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(Xtr_g, ytr,
          eval_set=[(Xval_g, yval)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    test_lgb_aug += m.predict_proba(X_test_geo)[:, 1] / 5

# Also use KNN on augmented set
aug_norm = normalize(aug_coords)
knn_aug = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
knn_aug.fit(aug_norm, aug_labels)
test_knn_aug = knn_aug.predict_proba(test_norm)[:, 1]

test_spatial_aug = 0.7 * test_lgb_aug + 0.3 * test_knn_aug

# Combine sns53 soft + augmented spatial
# Use sns53 as anchor, correct using augmented spatial
W_C_sns = 0.75
W_C_spa = 0.25
soft_blend_C = W_C_sns * sns53_soft + W_C_spa * test_spatial_aug
soft_pred_C = (soft_blend_C > 0.5).astype(int)
print(f"Approach C positive rate: {soft_pred_C.mean():.4f}")
print(f"vs sns53: {(soft_pred_C != sns53_labels).sum()} flips")

# ─── Approach D: Confidence-based flipping ───────────────────────────────────
print("\n=== APPROACH D: Confidence-based flipping ===")
# Start from sns53 predictions; flip only where augmented spatial model is extremely confident

# Compute confidence thresholds
flip_threshold = 0.85  # spatial must be very confident

flip_to_1 = (sns53_labels == 0) & (test_spatial_aug > flip_threshold)
flip_to_0 = (sns53_labels == 1) & (test_spatial_aug < (1 - flip_threshold))

pred_D = sns53_labels.copy()
pred_D[flip_to_1] = 1
pred_D[flip_to_0] = 0

print(f"Flipped 0→1: {flip_to_1.sum()}")
print(f"Flipped 1→0: {flip_to_0.sum()}")
print(f"Approach D positive rate: {pred_D.mean():.4f}")

# ─── CV simulation with noisy labels ─────────────────────────────────────────
print("\n\n=== CV SIMULATION: noisy pseudo-label quality test ===")
# Simulate having imperfect pseudo-labels (6.6% noise = sns53-like quality)
NOISE_RATE = 0.066

def simulate_noisy_cv(noise_rate, n_repeat=5, use_augmentation=True):
    """Simulate CV where test labels are noisy. Return mean accuracy."""
    all_accs = []
    skf_sim = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)

    for rep in range(n_repeat):
        rng = np.random.RandomState(rep * 17)
        # Introduce noise into training labels
        noisy_labels = y_train.copy().astype(float)
        flip_idx = rng.choice(len(noisy_labels), size=int(noise_rate * len(noisy_labels)), replace=False)
        noisy_labels[flip_idx] = 1 - noisy_labels[flip_idx]
        noisy_labels = noisy_labels.astype(int)

        fold_accs = []
        for tr_idx, val_idx in skf_sim.split(train_coords, y_train):
            Xtr_g = X_train_geo.iloc[tr_idx]
            Xval_g = X_train_geo.iloc[val_idx]
            ytr_noisy = noisy_labels[tr_idx]
            yval_true = y_train[val_idx]  # evaluate on TRUE labels

            m = lgb.LGBMClassifier(
                objective='binary', n_estimators=300, learning_rate=0.03,
                num_leaves=31, verbose=-1, random_state=42)
            m.fit(Xtr_g, ytr_noisy)
            preds = (m.predict_proba(Xval_g)[:, 1] > 0.5).astype(int)
            fold_accs.append(accuracy_score(yval_true, preds))

        all_accs.append(np.mean(fold_accs))

    return np.mean(all_accs), np.std(all_accs)

print("Training noisy CV (may take a moment)...")
noisy_mean, noisy_std = simulate_noisy_cv(NOISE_RATE, n_repeat=3)
print(f"Noisy pseudo-label CV accuracy: {noisy_mean:.4f} ± {noisy_std:.4f}")
print(f"Clean spatial OOF accuracy: {spatial_oof_acc:.4f}")

# ─── Approach E: Stacked ensemble (meta-learner) ─────────────────────────────
print("\n=== APPROACH E: Stacked meta-learner (final) ===")
# We have 3 signal sources per test point:
#   - sns53_soft (calibrated hard labels from 0.934 scorer)
#   - test_spatial (spatial ensemble trained on training data)
#   - test_spatial_aug (spatial retrained with pseudo-labels)
# Combine them optimally

# Sweep weights (sns53 vs spatial_aug) and thresholds
best_sim_acc = 0
best_w_ratio = None
best_thresh  = 0.5
best_blend   = None

for w_sns in np.arange(0.50, 0.96, 0.05):
    w_spa = 1.0 - w_sns
    blend = w_sns * sns53_soft + w_spa * test_spatial_aug
    for thresh in np.arange(0.40, 0.65, 0.02):
        preds = (blend > thresh).astype(int)
        # Proxy quality: use OOF spatial model calibration + sns53 agreement
        # Higher spatial confidence on AGREED points = better signal
        agree_high_conf = (preds == sns53_labels) & (test_spatial_aug > 0.65)
        disagree_low_conf = (preds != sns53_labels) & (test_spatial_aug < 0.55)
        sim_score = agree_high_conf.mean() - disagree_low_conf.mean()
        if sim_score > best_sim_acc:
            best_sim_acc = sim_score
            best_w_ratio = (w_sns, w_spa)
            best_thresh  = thresh
            best_blend   = blend.copy()

w_sns_best, w_spa_best = best_w_ratio
pred_E = (best_blend > best_thresh).astype(int)
print(f"Best weights: sns53={w_sns_best:.2f}, spatial_aug={w_spa_best:.2f}, threshold={best_thresh:.2f}")
print(f"Approach E positive rate: {pred_E.mean():.4f}")
print(f"vs sns53: {(pred_E != sns53_labels).sum()} flips")

# ─── Summary of all approaches ────────────────────────────────────────────────
print("\n\n=== APPROACH SUMMARY ===")
print(f"{'Approach':<20} {'Pos Rate':>10} {'Flips vs sns53':>15}")
print("-" * 50)
for name, preds in [
    ("A: Majority Vote",   majority),
    ("B: Soft Vote",       soft_pred_B),
    ("C: Semi-supervised", soft_pred_C),
    ("D: Conf Flip",       pred_D),
    ("E: Stacked Blend",   pred_E),
    ("sns53 (baseline)",   sns53_labels),
]:
    flips = (preds != sns53_labels).sum()
    print(f"{name:<20} {preds.mean():>10.4f} {flips:>15}")

# ─── Choose best approach ─────────────────────────────────────────────────────
print("\n\n=== SELECTING BEST APPROACH ===")

# Key reasoning:
# - sns53 scored 0.934, so it's very accurate
# - Our spatial model scored ~0.78-0.88 OOF, so it's weaker overall
# - But the two model types have DIFFERENT error patterns (climate vs geo)
# - Ideal: correct sns53's errors using spatial model's high-confidence signals
# - Approach C (semi-supervised) gives us a spatial model trained on more data
# - Approach D (flip only at >85% confidence) is most conservative
# - Approach E tries to optimally blend

# Let's compare Approaches C, D, E using OOF spatial accuracy as a guide
# The best approach conservatively flips only the most confident disagreements

# Let's also compute a high-quality semi-supervised blend:
# Start with sns53 and ONLY flip when:
# (a) augmented spatial is VERY confident (>0.85)
# (b) The original spatial also agrees with augmented spatial

FLIP_THRESH = 0.85
combined_spatial_for_flip = 0.6 * test_spatial_aug + 0.4 * test_spatial  # both agree

flip_up   = (sns53_labels == 0) & (combined_spatial_for_flip > FLIP_THRESH)
flip_down = (sns53_labels == 1) & (combined_spatial_for_flip < (1 - FLIP_THRESH))

pred_final = sns53_labels.copy()
pred_final[flip_up]   = 1
pred_final[flip_down] = 0

print(f"Final approach: confident spatial flip on sns53")
print(f"  Flipped 0→1: {flip_up.sum()}")
print(f"  Flipped 1→0: {flip_down.sum()}")
print(f"  Total flips: {flip_up.sum() + flip_down.sum()}")
print(f"  Final positive rate: {pred_final.mean():.4f}")

# Also build the best E blend as alternative
print(f"\nApproach E: {(pred_E != sns53_labels).sum()} flips, pos_rate={pred_E.mean():.4f}")

# DECIDE: use Approach E if it's making only a few targeted flips;
# otherwise use conservative flip approach
# We pick the blend that preserves the most of sns53 while using spatial corrections
# Approach C with moderate blending is likely the best

# Final selection: use soft blend Approach C (sns53=0.75, spatial_aug=0.25)
# but with threshold optimisation
print("\n=== Building submission_beat934.csv ===")
# Re-do Approach C with refined threshold search
best_pred_final = None
best_score_proxy = -1

for w_s in [0.70, 0.75, 0.80, 0.85, 0.88, 0.90]:
    w_sp = 1.0 - w_s
    blend = w_s * sns53_soft + w_sp * test_spatial_aug
    for thresh in np.arange(0.42, 0.62, 0.01):
        preds = (blend > thresh).astype(int)
        # Proxy: prefer positive rate close to training positive rate (0.6)
        # but weighted toward sns53 positive rate (which scores 0.934)
        target_pos = 0.934 * sns53_labels.mean() + (1 - 0.934) * y_train.mean()
        pos_rate_penalty = abs(preds.mean() - target_pos)

        # Prefer small number of flips from sns53 only where spatial is confident
        flips = (preds != sns53_labels).sum()
        confident_flips = 0
        for fi in np.where(preds != sns53_labels)[0]:
            if preds[fi] == 1 and test_spatial_aug[fi] > 0.70:
                confident_flips += 1
            elif preds[fi] == 0 and test_spatial_aug[fi] < 0.30:
                confident_flips += 1

        confident_ratio = confident_flips / max(flips, 1)
        score = confident_ratio - 0.5 * pos_rate_penalty - 0.001 * flips

        if score > best_score_proxy:
            best_score_proxy = score
            best_pred_final = preds.copy()
            best_config = (w_s, w_sp, thresh, flips, confident_flips)

w_s, w_sp, thresh, flips, conf_flips = best_config
print(f"Best config: w_sns53={w_s:.2f}, w_spatial={w_sp:.2f}, threshold={thresh:.2f}")
print(f"  Flips from sns53: {flips} ({conf_flips} confident)")
print(f"  Final positive rate: {best_pred_final.mean():.4f}")

# ─── Save submissions ─────────────────────────────────────────────────────────
submission = pd.DataFrame({'ID': test['ID'], 'Target': best_pred_final})
submission.to_csv(BASE + 'submission_beat934.csv', index=False)
print(f"\nSaved: submission_beat934.csv")
print(submission.head(10))

# Also save approach D (conservative flip) as backup
sub_D = pd.DataFrame({'ID': test['ID'], 'Target': pred_D})
sub_D.to_csv(BASE + 'submission_conf_flip.csv', index=False)
print(f"Saved: submission_conf_flip.csv")

# And approach E
sub_E = pd.DataFrame({'ID': test['ID'], 'Target': pred_E})
sub_E.to_csv(BASE + 'submission_stacked.csv', index=False)
print(f"Saved: submission_stacked.csv")

print("\n=== DONE ===")
print(f"submission_beat934.csv: {best_pred_final.sum()} positive, {(best_pred_final==0).sum()} negative")
print(f"submission_conf_flip.csv: {pred_D.sum()} positive, {(pred_D==0).sum()} negative")
print(f"submission_stacked.csv: {pred_E.sum()} positive, {(pred_E==0).sum()} negative")
