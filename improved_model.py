"""
Improved biodiversity prediction model — comprehensive spatial ensemble.

Approach: Since no climate data is available, we maximize information from
lat/lon through multiple complementary spatial methods:
  1. KNN classifier (best k by CV)
  2. SVM with RBF kernel (learned spatial boundary)
  3. Random Fourier Features + Logistic Regression (kernel approximation)
  4. KDE habitat suitability ratio
  5. LightGBM with rich polynomial + directional geo features
  6. Weighted ensemble blending with threshold optimization
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import lightgbm as lgb


# ── Data ────────────────────────────────────────────────────────────────────────
train = pd.read_csv('Training_Data.csv')
test  = pd.read_csv('Test.csv')

train_coords = train[['Latitude', 'Longitude']].values
test_coords  = test[['Latitude', 'Longitude']].values
y_train      = train['Occurrence Status'].values

pos_mask = y_train == 1
neg_mask = y_train == 0
pos_coords = train_coords[pos_mask]
neg_coords = train_coords[neg_mask]

print(f"Train: {len(train)} (pos={pos_mask.sum()}, neg={neg_mask.sum()})")
print(f"Test: {len(test)}")

# Normalize coordinates (used by all models)
lat_m, lat_s = train_coords[:,0].mean(), train_coords[:,0].std()
lon_m, lon_s = train_coords[:,1].mean(), train_coords[:,1].std()

def normalize(coords):
    return np.column_stack([
        (coords[:,0] - lat_m) / lat_s,
        (coords[:,1] - lon_m) / lon_s,
    ])

train_norm = normalize(train_coords)
test_norm  = normalize(test_coords)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Feature builder ──────────────────────────────────────────────────────────────
def build_geo_features(coords):
    lat = coords[:, 0]
    lon = coords[:, 1]
    ln  = (lat - lat_m) / lat_s
    lo  = (lon - lon_m) / lon_s
    # distance to known positive / negative centroids
    d_pc = np.sqrt((lat - pos_coords[:,0].mean())**2 + (lon - pos_coords[:,1].mean())**2)
    d_nc = np.sqrt((lat - neg_coords[:,0].mean())**2 + (lon - neg_coords[:,1].mean())**2)
    # Distance to SE Australia corners / reference points
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
        'ln2_lo':  ln**2*lo, 'ln_lo2':  ln*lo**2,
        'ln3_lo':  ln**3*lo, 'ln_lo3':  ln*lo**3,
        'ln2_lo2': ln**2*lo**2,
        'dist': np.sqrt(ln**2 + lo**2),
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

# ── Model definitions ────────────────────────────────────────────────────────────

# 1. KNN — tune k
print("\n=== KNN tuning ===")
best_k, best_k_acc = 5, 0
for k in [3, 5, 7, 10, 15, 20, 30]:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    acc = cross_val_score(knn, train_norm, y_train, cv=skf, scoring='accuracy').mean()
    print(f"  k={k:3d}: {acc:.4f}")
    if acc > best_k_acc:
        best_k_acc, best_k = acc, k
print(f"Best k={best_k} ({best_k_acc:.4f})")

# 2. SVM with RBF kernel — tune C and gamma
print("\n=== SVM RBF tuning ===")
best_svm_acc, best_svm_params = 0, (1.0, 1.0)
for C in [0.5, 1.0, 5.0, 10.0, 50.0]:
    for gamma in [0.1, 0.5, 1.0, 2.0, 5.0]:
        svm = SVC(C=C, gamma=gamma, kernel='rbf', probability=True)
        acc = cross_val_score(svm, train_norm, y_train, cv=skf, scoring='accuracy').mean()
        if acc > best_svm_acc:
            best_svm_acc, best_svm_params = acc, (C, gamma)
print(f"Best SVM: C={best_svm_params[0]}, gamma={best_svm_params[1]} ({best_svm_acc:.4f})")

# 3. Random Fourier Features + Logistic Regression — tune n_components, gamma, C
print("\n=== RFF + LR tuning ===")
best_rff_acc, best_rff_params = 0, (500, 1.0, 1.0)
for nc in [200, 500, 1000]:
    for gamma in [0.5, 1.0, 2.0, 5.0]:
        for C in [0.1, 0.5, 1.0]:
            pipe = Pipeline([
                ('rff', RBFSampler(gamma=gamma, n_components=nc, random_state=42)),
                ('lr', LogisticRegression(C=C, max_iter=500, solver='saga')),
            ])
            acc = cross_val_score(pipe, train_norm, y_train, cv=skf, scoring='accuracy').mean()
            if acc > best_rff_acc:
                best_rff_acc, best_rff_params = acc, (nc, gamma, C)
print(f"Best RFF+LR: n={best_rff_params[0]}, gamma={best_rff_params[1]}, C={best_rff_params[2]} ({best_rff_acc:.4f})")

# 4. KDE ratio
def kde_oof():
    oof = np.zeros(len(train_coords))
    for tr_idx, val_idx in skf.split(train_coords, y_train):
        Xtr, Xval = train_coords[tr_idx], train_coords[val_idx]
        ytr = y_train[tr_idx]
        p_tr = Xtr[ytr==1]; n_tr = Xtr[ytr==0]
        np_, nn_ = (ytr==1).sum(), (ytr==0).sum()
        kp = gaussian_kde(p_tr.T, bw_method='silverman')
        kn = gaussian_kde(n_tr.T, bw_method='silverman')
        pv = kp(Xval.T)*np_; nv = kn(Xval.T)*nn_
        oof[val_idx] = pv / (pv + nv + 1e-12)
    return oof

kde_oof_pred = kde_oof()
kde_acc = accuracy_score(y_train, (kde_oof_pred > 0.5).astype(int))
print(f"\nKDE acc: {kde_acc:.4f}")

# 5. LightGBM geo features — tune
print("\n=== LGB geo tuning ===")
lgb_params = dict(
    objective='binary', metric='binary_logloss',
    n_estimators=500, learning_rate=0.03,
    num_leaves=63, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=0.5,
    verbose=-1, random_state=42,
)
lgb_cv = []
for tr_idx, val_idx in skf.split(X_train_geo, y_train):
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(X_train_geo.iloc[tr_idx], y_train[tr_idx],
          eval_set=[(X_train_geo.iloc[val_idx], y_train[val_idx])],
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)])
    preds = (m.predict_proba(X_train_geo.iloc[val_idx])[:, 1] > 0.5).astype(int)
    lgb_cv.append(accuracy_score(y_train[val_idx], preds))
lgb_acc = np.mean(lgb_cv)
print(f"LGB geo acc: {lgb_acc:.4f}")


# ── Full OOF prediction pass ─────────────────────────────────────────────────────
print("\n=== Building OOF predictions ===")
oof = {
    'knn': np.zeros(len(train_coords)),
    'svm': np.zeros(len(train_coords)),
    'rff': np.zeros(len(train_coords)),
    'lgb': np.zeros(len(train_coords)),
    'kde': kde_oof_pred.copy(),
}
test_proba = {k: np.zeros(len(test_coords)) for k in oof}

knn_model     = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
svm_model     = SVC(C=best_svm_params[0], gamma=best_svm_params[1], kernel='rbf', probability=True)
rff_lr_model  = Pipeline([
    ('rff', RBFSampler(gamma=best_rff_params[1], n_components=best_rff_params[0], random_state=42)),
    ('lr', LogisticRegression(C=best_rff_params[2], max_iter=500, solver='saga')),
])

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_coords, y_train)):
    print(f"  Fold {fold+1}/5...", end=' ', flush=True)
    Xtr_n, Xval_n = train_norm[tr_idx], train_norm[val_idx]
    Xtr_g, Xval_g = X_train_geo.iloc[tr_idx], X_train_geo.iloc[val_idx]
    ytr = y_train[tr_idx]

    knn_model.fit(Xtr_n, ytr)
    oof['knn'][val_idx] = knn_model.predict_proba(Xval_n)[:, 1]
    test_proba['knn'] += knn_model.predict_proba(test_norm)[:, 1] / 5

    svm_model.fit(Xtr_n, ytr)
    oof['svm'][val_idx] = svm_model.predict_proba(Xval_n)[:, 1]
    test_proba['svm'] += svm_model.predict_proba(test_norm)[:, 1] / 5

    rff_lr_model.fit(Xtr_n, ytr)
    oof['rff'][val_idx] = rff_lr_model.predict_proba(Xval_n)[:, 1]
    test_proba['rff'] += rff_lr_model.predict_proba(test_norm)[:, 1] / 5

    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(Xtr_g, ytr,
          eval_set=[(Xval_g, y_train[val_idx])],
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)])
    oof['lgb'][val_idx] = m.predict_proba(Xval_g)[:, 1]
    test_proba['lgb'] += m.predict_proba(X_test_geo)[:, 1] / 5

    # KDE
    pos_tr = train_coords[tr_idx][ytr==1]; neg_tr = train_coords[tr_idx][ytr==0]
    np_, nn_ = (ytr==1).sum(), (ytr==0).sum()
    kp = gaussian_kde(pos_tr.T, bw_method='silverman')
    kn = gaussian_kde(neg_tr.T, bw_method='silverman')
    pv_t = kp(test_coords.T)*np_; nv_t = kn(test_coords.T)*nn_
    test_proba['kde'] += (pv_t / (pv_t + nv_t + 1e-12)) / 5

    print(f"knn={accuracy_score(y_train[val_idx],(oof['knn'][val_idx]>0.5).astype(int)):.4f} "
          f"svm={accuracy_score(y_train[val_idx],(oof['svm'][val_idx]>0.5).astype(int)):.4f} "
          f"rff={accuracy_score(y_train[val_idx],(oof['rff'][val_idx]>0.5).astype(int)):.4f} "
          f"lgb={accuracy_score(y_train[val_idx],(oof['lgb'][val_idx]>0.5).astype(int)):.4f}")

print("\nFinal OOF accuracies:")
for name, pred in oof.items():
    print(f"  {name}: {accuracy_score(y_train, (pred>0.5).astype(int)):.4f}")


# ── Ensemble weight optimization ─────────────────────────────────────────────────
print("\n=== Optimizing ensemble ===")
keys = list(oof.keys())
oof_arr = np.array([oof[k] for k in keys])       # (5, n_train)
test_arr = np.array([test_proba[k] for k in keys]) # (5, n_test)

best_acc, best_params = 0, None
for w0 in np.arange(0.0, 1.01, 0.1):
  for w1 in np.arange(0.0, 1.01-w0, 0.1):
    for w2 in np.arange(0.0, 1.01-w0-w1, 0.1):
      for w3 in np.arange(0.0, 1.01-w0-w1-w2, 0.1):
        w4 = round(1.0-w0-w1-w2-w3, 2)
        if w4 < 0: continue
        W = np.array([w0,w1,w2,w3,w4])
        blend = W @ oof_arr
        for thresh in np.arange(0.35, 0.65, 0.02):
            acc = accuracy_score(y_train, (blend > thresh).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_params = (W.copy(), thresh)

W_best, thresh_best = best_params
print(f"Best OOF accuracy: {best_acc:.6f}")
print(f"Weights: {dict(zip(keys, W_best.round(2).tolist()))}")
print(f"Threshold: {thresh_best:.2f}")


# ── Final predictions ────────────────────────────────────────────────────────────
test_blend = W_best @ test_arr
test_preds = (test_blend > thresh_best).astype(int)

print(f"\nTest: {test_preds.sum()} positive ({test_preds.mean():.3f}), "
      f"{(test_preds==0).sum()} negative")


# ── Save ────────────────────────────────────────────────────────────────────────
submission = pd.DataFrame({'ID': test['ID'], 'Target': test_preds})
submission.to_csv('submission_improved.csv', index=False)
print("\nSaved: submission_improved.csv")
print(submission.head(10).to_string())
