from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "05_strong_classical_baseline.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.x"},
}

nb["cells"] = [
    md(
        """
        # 05. Strong Classical Baseline cho Speech Emotion Recognition

        Notebook này nâng cấp baseline MFCC/classical ML trước khi chuyển sang CNN, LSTM, SSL embedding hoặc emotion2vec.

        Mục tiêu chính:

        - So sánh **strict speaker-aware split** hiện tại với **paper-comparable random split**.
        - Tune các mô hình classical: Logistic Regression, SVM RBF, Random Forest, Extra Trees, KNN.
        - Thử feature selection nhẹ bằng `SelectKBest` cho nhóm model cần scale.
        - Tạo ensemble soft-voting từ các model mạnh nhất.
        - Xuất đầy đủ report, prediction, hình visualize và file ZIP để tải về.

        Vì notebook dùng artifact `baseline_features.npz`, Kaggle chỉ cần dataset/repo có `ser_processed/metadata.csv` và `ser_processed/baseline_features.npz`; không cần upload `audio_16k`.
        """
    ),
    md(
        """
        ## 1. Vì Sao Cần Strong Baseline?

        Một số paper báo kết quả rất cao, nhưng thường không cùng điều kiện đánh giá:

        - [Novais et al. - Emotion Classification from Speech by an Ensemble Strategy](https://dl.acm.org/doi/fullHtml/10.1145/3563137.3563170): dùng aggregator/ensemble, đạt khoảng 63.64% trên tập gộp RAVDESS + CREMA-D + SAVEE + TESS.
        - [Novais dissertation PDF](https://sapientia.ualg.pt/bitstreams/5679c642-b258-4d75-b9a8-d8ad820feba3/download): báo cáo baseline và ensemble chi tiết cho multi-dataset.
        - [Speech Emotion Recognition Using ML Models and Audio Features](https://computing.louisiana.edu/sites/computing/files/Speech_Emotion_Recognition_Using_ML_Models_and_Audio_Features.pdf): SVM đạt 90.65% trên EmoDB nhờ MFCC + pitch + RMS + silence removal + tuning, nhưng đây là single-dataset sạch hơn.
        - [MFMC/MFCC + SVM summary](https://arxiv.org/html/2507.03251v2): modified MFCC/MFMC + SVM đạt 64.31% trên RAVDESS.

        Vì vậy notebook này báo cáo hai kiểu:

        - **Strict split:** dùng split từ Data Processing, tôn trọng speaker/group split. Đây là kết quả chính nên đưa vào báo cáo.
        - **Random split:** dùng stratified random split để có mốc so sánh gần hơn với nhiều Kaggle/paper baseline.
        """
    ),
    md(
        """
        ## 2. Cấu Hình

        Cell này tự tìm project/dataset trên Kaggle, Colab hoặc máy local.
        """
    ),
    code(
        """
        from pathlib import Path
        import json
        import math
        import os
        import shutil
        import time
        import warnings

        import joblib
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        from IPython.display import display

        from sklearn.base import clone
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )
        from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.svm import SVC

        warnings.filterwarnings("ignore")
        sns.set_theme(style="whitegrid", context="notebook")

        RANDOM_STATE = 42
        N_JOBS = -1
        QUICK_RUN = False

        # Tuning nhỏ nhưng đủ để notebook chạy được trên Kaggle/Colab CPU.
        # Tăng N_ITER_SEARCH nếu muốn benchmark kỹ hơn.
        N_ITER_SEARCH = 4
        CV_FOLDS = 2

        LABEL_ORDER = ["neutral", "happy", "sad", "angry", "fear", "disgust"]
        """
    ),
    code(
        """
        def find_project_root():
            candidates = []

            # Kaggle input/working patterns
            for base in [Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()]:
                if base.exists():
                    candidates.append(base)
                    candidates.extend([p for p in base.glob("**") if p.is_dir() and len(p.parts) - len(base.parts) <= 4])

            # Colab / local fallback
            colab_root = Path("/content/drive/MyDrive/Speech_Project")
            if colab_root.exists():
                candidates.insert(0, colab_root)

            cwd = Path.cwd().resolve()
            candidates.extend([cwd, *cwd.parents])

            seen = set()
            for candidate in candidates:
                candidate = candidate.resolve()
                if candidate in seen:
                    continue
                seen.add(candidate)

                direct = candidate / "ser_processed"
                nested = candidate / "01&02_Data_and_DataProcessing" / "ser_processed"

                if (direct / "metadata.csv").exists() and (direct / "baseline_features.npz").exists():
                    return candidate, direct
                if (nested / "metadata.csv").exists() and (nested / "baseline_features.npz").exists():
                    return candidate, nested

            raise FileNotFoundError(
                "Không tìm thấy ser_processed/metadata.csv và baseline_features.npz. "
                "Hãy upload repo/dataset đã có ser_processed lên Kaggle."
            )


        PROJECT_ROOT, PROCESSED_DIR = find_project_root()
        NOTEBOOK_DIR = Path.cwd().resolve()
        OUTPUT_DIR = NOTEBOOK_DIR / "outputs"
        FIG_DIR = OUTPUT_DIR / "figures"
        REPORT_DIR = OUTPUT_DIR / "reports"
        MODEL_DIR = OUTPUT_DIR / "models"
        PRED_DIR = OUTPUT_DIR / "predictions"

        for d in [OUTPUT_DIR, FIG_DIR, REPORT_DIR, MODEL_DIR, PRED_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        print("PROJECT_ROOT:", PROJECT_ROOT)
        print("PROCESSED_DIR:", PROCESSED_DIR)
        print("OUTPUT_DIR:", OUTPUT_DIR)
        """
    ),
    md(
        """
        ## 3. Load Artifact Và Kiểm Tra Dữ Liệu

        `baseline_features.npz` là vector 248 chiều được trích ở Data Processing:

        - MFCC 40 hệ số.
        - Delta MFCC.
        - Delta-delta MFCC.
        - RMS.
        - Zero-crossing rate.
        - Spectral centroid.
        - Spectral bandwidth.

        Mỗi nhóm feature được lấy mean và std theo thời gian.
        """
    ),
    code(
        """
        metadata = pd.read_csv(PROCESSED_DIR / "metadata.csv")
        features = np.load(PROCESSED_DIR / "baseline_features.npz", allow_pickle=True)

        X_raw = features["X"].astype(np.float32)
        y_text = features["y"].astype(str)
        sample_ids = features["sample_id"].astype(str)
        original_split = features["split"].astype(str)

        le = LabelEncoder()
        le.fit(LABEL_ORDER)
        y = le.transform(y_text)

        print("metadata:", metadata.shape)
        print("X_raw:", X_raw.shape)
        print("labels:", dict(zip(*np.unique(y_text, return_counts=True))))
        print("split:", dict(zip(*np.unique(original_split, return_counts=True))))

        assert len(metadata) == X_raw.shape[0] == len(y_text) == len(sample_ids)
        assert set(y_text).issubset(set(LABEL_ORDER))
        display(metadata.head())
        """
    ),
    code(
        """
        checks = {
            "n_samples": len(metadata),
            "feature_dim": X_raw.shape[1],
            "missing_label": int(pd.isna(y_text).sum()),
            "duplicate_sample_id": int(pd.Series(sample_ids).duplicated().sum()),
            "duplicate_dataset_source": int(metadata.duplicated(["dataset", "source_filename"]).sum())
                if {"dataset", "source_filename"}.issubset(metadata.columns) else None,
            "has_train": bool(np.any(original_split == "train")),
            "has_test": bool(np.any(original_split == "test")),
        }
        pd.Series(checks).to_frame("value")
        """
    ),
    code(
        """
        split_dataset = pd.crosstab(metadata["dataset"], original_split)
        split_emotion = pd.crosstab(pd.Series(y_text, name="emotion"), original_split)

        display(split_dataset)
        display(split_emotion)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        split_dataset.plot(kind="bar", stacked=True, ax=axes[0])
        axes[0].set_title("Strict split by dataset")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Samples")
        split_emotion.loc[LABEL_ORDER].plot(kind="bar", stacked=True, ax=axes[1])
        axes[1].set_title("Strict split by emotion")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Samples")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "strict_split_distribution.png", dpi=160)
        plt.show()
        """
    ),
    md(
        """
        ## 4. Tạo Hai Scenario Đánh Giá

        - `strict`: dùng split có sẵn từ Data Processing.
        - `random`: stratified random split 70/15/15 để so với nhiều baseline trên paper/Kaggle.
        """
    ),
    code(
        """
        def make_strict_split():
            train_idx = np.where(original_split == "train")[0]
            val_idx = np.where(original_split == "validation")[0]
            test_idx = np.where(original_split == "test")[0]
            if len(val_idx) == 0:
                train_idx, val_idx = train_test_split(
                    train_idx,
                    test_size=0.12,
                    random_state=RANDOM_STATE,
                    stratify=y[train_idx],
                )
            return {
                "name": "strict_speaker_aware",
                "train": train_idx,
                "validation": val_idx,
                "test": test_idx,
                "description": "Original split from Data Processing, group-aware by speaker_id where available.",
            }


        def make_random_split():
            all_idx = np.arange(len(y))
            train_idx, temp_idx = train_test_split(
                all_idx,
                test_size=0.30,
                random_state=RANDOM_STATE,
                stratify=y,
            )
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=0.50,
                random_state=RANDOM_STATE,
                stratify=y[temp_idx],
            )
            return {
                "name": "paper_comparable_random",
                "train": train_idx,
                "validation": val_idx,
                "test": test_idx,
                "description": "Stratified random 70/15/15 split. Useful for comparison with many papers, but less strict.",
            }


        scenarios = [make_strict_split(), make_random_split()]
        for s in scenarios:
            print("\\n", s["name"])
            for split_name in ["train", "validation", "test"]:
                idx = s[split_name]
                print(split_name, len(idx), dict(zip(le.classes_, np.bincount(y[idx], minlength=len(le.classes_)))))
        """
    ),
    md(
        """
        ## 5. Model Zoo: Classical ML + Tuning

        Tất cả model được tune bằng cross-validation trên train set. Test set chỉ dùng một lần để báo cáo cuối.

        Model:

        - Dummy majority baseline.
        - Logistic Regression.
        - SVM RBF.
        - Random Forest.
        - Extra Trees.
        - KNN.
        - Soft Voting ensemble từ các model có `predict_proba`.
        """
    ),
    code(
        """
        def make_model_specs():
            k_all = X_raw.shape[1]
            return {
                "dummy_majority": {
                    "estimator": DummyClassifier(strategy="most_frequent"),
                    "params": {},
                    "search": False,
                },
                "logistic_regression_tuned": {
                    "estimator": Pipeline([
                        ("scaler", StandardScaler()),
                        ("select", SelectKBest(score_func=f_classif, k=min(160, k_all))),
                        ("clf", LogisticRegression(max_iter=2500, solver="saga", random_state=RANDOM_STATE)),
                    ]),
                    "params": {
                        "select__k": [80, 120, 160, "all"],
                        "clf__C": [0.05, 0.1, 0.3, 1.0, 3.0],
                        "clf__penalty": ["l2"],
                        "clf__class_weight": [None, "balanced"],
                    },
                    "search": True,
                },
                "svm_rbf_tuned": {
                    "estimator": Pipeline([
                        ("scaler", StandardScaler()),
                        ("select", SelectKBest(score_func=f_classif, k=min(160, k_all))),
                        ("clf", SVC(kernel="rbf", probability=False, random_state=RANDOM_STATE)),
                    ]),
                    "params": {
                        "select__k": [120, 160, "all"],
                        "clf__C": [1.0, 3.0, 10.0],
                        "clf__gamma": ["scale", 0.01, 0.003],
                        "clf__class_weight": [None, "balanced"],
                    },
                    "search": True,
                },
                "random_forest_tuned": {
                    "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS),
                    "params": {
                        "n_estimators": [200, 400],
                        "max_depth": [None, 18, 32],
                        "min_samples_leaf": [1, 2, 4],
                        "max_features": ["sqrt", "log2", 0.4],
                        "class_weight": [None, "balanced_subsample"],
                    },
                    "search": True,
                },
                "extra_trees_tuned": {
                    "estimator": ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS),
                    "params": {
                        "n_estimators": [200, 400],
                        "max_depth": [None, 18, 32],
                        "min_samples_leaf": [1, 2, 4],
                        "max_features": ["sqrt", "log2", 0.4],
                        "class_weight": [None, "balanced"],
                    },
                    "search": True,
                },
                "knn_tuned": {
                    "estimator": Pipeline([
                        ("scaler", StandardScaler()),
                        ("select", SelectKBest(score_func=f_classif, k=min(160, k_all))),
                        ("clf", KNeighborsClassifier()),
                    ]),
                    "params": {
                        "select__k": [80, 120, 160, "all"],
                        "clf__n_neighbors": [3, 5, 9, 15, 25],
                        "clf__weights": ["uniform", "distance"],
                        "clf__metric": ["minkowski", "manhattan"],
                    },
                    "search": True,
                },
            }


        def fit_estimator(name, spec, X_train, y_train):
            start = time.perf_counter()
            if spec["search"]:
                cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                n_iter = min(N_ITER_SEARCH, math.prod([len(v) for v in spec["params"].values()]))
                search = RandomizedSearchCV(
                    estimator=spec["estimator"],
                    param_distributions=spec["params"],
                    n_iter=n_iter,
                    scoring="f1_macro",
                    cv=cv,
                    n_jobs=N_JOBS,
                    random_state=RANDOM_STATE,
                    refit=True,
                    verbose=0,
                )
                search.fit(X_train, y_train)
                estimator = search.best_estimator_
                best_params = search.best_params_
                cv_macro_f1 = float(search.best_score_)
            else:
                estimator = clone(spec["estimator"])
                estimator.fit(X_train, y_train)
                best_params = {}
                cv_macro_f1 = np.nan

            train_time = time.perf_counter() - start
            return estimator, best_params, cv_macro_f1, train_time


        def evaluate_estimator(estimator, scenario_name, model_name, split_name, indices, train_time, cv_macro_f1, best_params):
            X_eval = X_raw[indices]
            y_eval = y[indices]
            start = time.perf_counter()
            pred = estimator.predict(X_eval)
            inference_time = time.perf_counter() - start

            row = {
                "scenario": scenario_name,
                "model": model_name,
                "split": split_name,
                "n_samples": len(indices),
                "accuracy": accuracy_score(y_eval, pred),
                "macro_f1": f1_score(y_eval, pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(y_eval, pred, average="weighted", zero_division=0),
                "macro_precision": precision_score(y_eval, pred, average="macro", zero_division=0),
                "macro_recall": recall_score(y_eval, pred, average="macro", zero_division=0),
                "cv_macro_f1": cv_macro_f1,
                "train_time_sec": train_time,
                "inference_time_sec": inference_time,
                "inference_ms_per_sample": inference_time / max(len(indices), 1) * 1000,
                "best_params": json.dumps(best_params, ensure_ascii=False),
            }

            pred_df = metadata.iloc[indices].copy().reset_index(drop=True)
            pred_df["y_true"] = le.inverse_transform(y_eval)
            pred_df["y_pred"] = le.inverse_transform(pred)
            pred_df["correct"] = pred_df["y_true"] == pred_df["y_pred"]
            return row, pred_df, pred
        """
    ),
    code(
        """
        all_metrics = []
        all_predictions = {}
        fitted_models = {}
        best_params_records = []

        for scenario in scenarios:
            scenario_name = scenario["name"]
            print(f"\\n=== Scenario: {scenario_name} ===")
            X_train, y_train = X_raw[scenario["train"]], y[scenario["train"]]
            specs = make_model_specs()

            for model_name, spec in specs.items():
                print(f"Training {model_name} ...")
                estimator, best_params, cv_macro_f1, train_time = fit_estimator(model_name, spec, X_train, y_train)
                fitted_models[(scenario_name, model_name)] = estimator
                best_params_records.append({
                    "scenario": scenario_name,
                    "model": model_name,
                    "cv_macro_f1": cv_macro_f1,
                    "train_time_sec": train_time,
                    "best_params": json.dumps(best_params, ensure_ascii=False),
                })

                for split_name in ["validation", "test"]:
                    row, pred_df, _ = evaluate_estimator(
                        estimator,
                        scenario_name,
                        model_name,
                        split_name,
                        scenario[split_name],
                        train_time,
                        cv_macro_f1,
                        best_params,
                    )
                    all_metrics.append(row)
                    all_predictions[(scenario_name, model_name, split_name)] = pred_df
                    pred_df.to_csv(PRED_DIR / f"predictions_{scenario_name}_{model_name}_{split_name}.csv", index=False)

                joblib.dump(estimator, MODEL_DIR / f"{scenario_name}_{model_name}.pkl")

        metrics_df = pd.DataFrame(all_metrics)
        params_df = pd.DataFrame(best_params_records)
        metrics_df.to_csv(REPORT_DIR / "strong_classical_metrics.csv", index=False)
        params_df.to_csv(REPORT_DIR / "strong_classical_best_params.csv", index=False)

        display(metrics_df.sort_values(["scenario", "split", "macro_f1"], ascending=[True, True, False]))
        """
    ),
    md(
        """
        ## 6. Probability-Average Ensemble

        Ensemble được tạo sau khi đã tune model đơn. Notebook dùng trung bình xác suất từ các model đã fit để tránh refit nặng, tương đương ý tưởng soft-voting ở mức prediction.
        """
    ),
    code(
        """
        class ProbabilityAverageEnsemble:
            def __init__(self, members):
                self.members = members
                self.classes_ = np.arange(len(LABEL_ORDER))

            def predict_proba(self, X):
                probs = []
                for _, model in self.members:
                    p = model.predict_proba(X)
                    probs.append(p)
                return np.mean(probs, axis=0)

            def predict(self, X):
                return np.argmax(self.predict_proba(X), axis=1)


        def has_predict_proba(model):
            try:
                getattr(model, "predict_proba")
                return True
            except Exception:
                return False


        ensemble_metrics = []

        for scenario in scenarios:
            scenario_name = scenario["name"]
            val_rows = metrics_df[(metrics_df["scenario"] == scenario_name) & (metrics_df["split"] == "validation")]
            ranked_names = [
                m for m in val_rows.sort_values("macro_f1", ascending=False)["model"].tolist()
                if m not in ["dummy_majority"]
            ]

            members = []
            for model_name in ranked_names:
                est = fitted_models[(scenario_name, model_name)]
                if has_predict_proba(est):
                    members.append((model_name, est))
                if len(members) >= 4:
                    break

            if len(members) < 2:
                print(f"Skip ensemble for {scenario_name}: not enough probability models.")
                continue

            print(f"\\nProbability-average ensemble for {scenario_name}: {[n for n, _ in members]}")
            ensemble = ProbabilityAverageEnsemble(members)
            fitted_models[(scenario_name, "probability_average_ensemble")] = ensemble
            joblib.dump(ensemble, MODEL_DIR / f"{scenario_name}_probability_average_ensemble.pkl")

            for split_name in ["validation", "test"]:
                row, pred_df, _ = evaluate_estimator(
                    ensemble,
                    scenario_name,
                    "probability_average_ensemble",
                    split_name,
                    scenario[split_name],
                    train_time=0.0,
                    cv_macro_f1=np.nan,
                    best_params={"members": [name for name, _ in members]},
                )
                ensemble_metrics.append(row)
                all_predictions[(scenario_name, "probability_average_ensemble", split_name)] = pred_df
                pred_df.to_csv(PRED_DIR / f"predictions_{scenario_name}_probability_average_ensemble_{split_name}.csv", index=False)

        if ensemble_metrics:
            metrics_df = pd.concat([metrics_df, pd.DataFrame(ensemble_metrics)], ignore_index=True)
            metrics_df.to_csv(REPORT_DIR / "strong_classical_metrics.csv", index=False)

        display(metrics_df.sort_values(["scenario", "split", "macro_f1"], ascending=[True, True, False]))
        """
    ),
    md(
        """
        ## 7. Chọn Best Model Theo Validation

        Chọn model theo validation macro-F1. Test chỉ dùng để báo cáo cuối.
        """
    ),
    code(
        """
        best_rows = (
            metrics_df[metrics_df["split"] == "validation"]
            .sort_values(["scenario", "macro_f1", "accuracy"], ascending=[True, False, False])
            .groupby("scenario")
            .head(1)
            .reset_index(drop=True)
        )
        display(best_rows[["scenario", "model", "accuracy", "macro_f1", "weighted_f1", "best_params"]])

        test_best_rows = []
        for row in best_rows.itertuples(index=False):
            test_row = metrics_df[
                (metrics_df["scenario"] == row.scenario)
                & (metrics_df["model"] == row.model)
                & (metrics_df["split"] == "test")
            ].iloc[0]
            test_best_rows.append(test_row)

            best_estimator = fitted_models[(row.scenario, row.model)]
            joblib.dump(best_estimator, MODEL_DIR / f"BEST_{row.scenario}_{row.model}.pkl")

        best_test_df = pd.DataFrame(test_best_rows)
        display(best_test_df[["scenario", "model", "n_samples", "accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]])
        best_test_df.to_csv(REPORT_DIR / "best_models_test_summary.csv", index=False)
        """
    ),
    md(
        """
        ## 8. Visualize Kết Quả
        """
    ),
    code(
        """
        plot_df = metrics_df[metrics_df["split"] == "test"].copy()

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(data=plot_df, x="model", y="macro_f1", hue="scenario", ax=ax)
        ax.set_title("Test Macro-F1: strict split vs paper-comparable random split")
        ax.set_xlabel("")
        ax.set_ylabel("Macro-F1")
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylim(0, min(1.0, max(0.2, plot_df["macro_f1"].max() + 0.1)))
        plt.tight_layout()
        plt.savefig(FIG_DIR / "test_macro_f1_comparison.png", dpi=180)
        plt.show()

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(data=plot_df, x="model", y="accuracy", hue="scenario", ax=ax)
        ax.set_title("Test Accuracy: strict split vs paper-comparable random split")
        ax.set_xlabel("")
        ax.set_ylabel("Accuracy")
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylim(0, min(1.0, max(0.2, plot_df["accuracy"].max() + 0.1)))
        plt.tight_layout()
        plt.savefig(FIG_DIR / "test_accuracy_comparison.png", dpi=180)
        plt.show()
        """
    ),
    code(
        """
        for row in best_rows.itertuples(index=False):
            scenario_name = row.scenario
            model_name = row.model
            pred_df = all_predictions[(scenario_name, model_name, "test")]
            cm = confusion_matrix(pred_df["y_true"], pred_df["y_pred"], labels=LABEL_ORDER)
            cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER, ax=ax)
            ax.set_title(f"Normalized confusion matrix\\n{scenario_name} | {model_name}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"confusion_matrix_{scenario_name}_{model_name}.png", dpi=180)
            plt.show()
        """
    ),
    code(
        """
        per_class_records = []
        for row in best_rows.itertuples(index=False):
            scenario_name = row.scenario
            model_name = row.model
            pred_df = all_predictions[(scenario_name, model_name, "test")]
            report = classification_report(pred_df["y_true"], pred_df["y_pred"], labels=LABEL_ORDER, output_dict=True, zero_division=0)
            for emotion in LABEL_ORDER:
                per_class_records.append({
                    "scenario": scenario_name,
                    "model": model_name,
                    "emotion": emotion,
                    "precision": report[emotion]["precision"],
                    "recall": report[emotion]["recall"],
                    "f1_score": report[emotion]["f1-score"],
                    "support": report[emotion]["support"],
                })

        per_class_df = pd.DataFrame(per_class_records)
        per_class_df.to_csv(REPORT_DIR / "best_models_per_class_test.csv", index=False)
        display(per_class_df)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=per_class_df, x="emotion", y="f1_score", hue="scenario", order=LABEL_ORDER, ax=ax)
        ax.set_title("Best model per-class F1 on test")
        ax.set_xlabel("")
        ax.set_ylabel("F1-score")
        ax.set_ylim(0, min(1.0, per_class_df["f1_score"].max() + 0.12))
        plt.tight_layout()
        plt.savefig(FIG_DIR / "best_models_per_class_f1.png", dpi=180)
        plt.show()
        """
    ),
    code(
        """
        per_dataset_records = []
        if "dataset" in metadata.columns:
            for row in best_rows.itertuples(index=False):
                scenario_name = row.scenario
                model_name = row.model
                pred_df = all_predictions[(scenario_name, model_name, "test")]
                for dataset, part in pred_df.groupby("dataset"):
                    per_dataset_records.append({
                        "scenario": scenario_name,
                        "model": model_name,
                        "dataset": dataset,
                        "n_samples": len(part),
                        "accuracy": accuracy_score(part["y_true"], part["y_pred"]),
                        "macro_f1": f1_score(part["y_true"], part["y_pred"], labels=LABEL_ORDER, average="macro", zero_division=0),
                        "weighted_f1": f1_score(part["y_true"], part["y_pred"], average="weighted", zero_division=0),
                    })

            per_dataset_df = pd.DataFrame(per_dataset_records)
            per_dataset_df.to_csv(REPORT_DIR / "best_models_per_dataset_test.csv", index=False)
            display(per_dataset_df)

            fig, ax = plt.subplots(figsize=(9, 5))
            sns.barplot(data=per_dataset_df, x="dataset", y="macro_f1", hue="scenario", ax=ax)
            ax.set_title("Best model macro-F1 by dataset")
            ax.set_xlabel("")
            ax.set_ylabel("Macro-F1")
            ax.set_ylim(0, min(1.0, per_dataset_df["macro_f1"].max() + 0.12))
            plt.tight_layout()
            plt.savefig(FIG_DIR / "best_models_per_dataset_macro_f1.png", dpi=180)
            plt.show()
        """
    ),
    md(
        """
        ## 9. Error Analysis

        Bảng dưới cho biết các cặp nhãn dễ nhầm nhất.
        """
    ),
    code(
        """
        error_records = []
        for row in best_rows.itertuples(index=False):
            scenario_name = row.scenario
            model_name = row.model
            pred_df = all_predictions[(scenario_name, model_name, "test")]
            wrong = pred_df[~pred_df["correct"]].copy()
            crosstab = pd.crosstab(wrong["y_true"], wrong["y_pred"])
            crosstab.to_csv(REPORT_DIR / f"error_crosstab_{scenario_name}_{model_name}.csv")
            top_errors = (
                wrong.groupby(["y_true", "y_pred"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(12)
            )
            top_errors["scenario"] = scenario_name
            top_errors["model"] = model_name
            error_records.append(top_errors)

        error_df = pd.concat(error_records, ignore_index=True)
        error_df.to_csv(REPORT_DIR / "top_error_pairs.csv", index=False)
        display(error_df[["scenario", "model", "y_true", "y_pred", "count"]])
        """
    ),
    md(
        """
        ## 10. Kết Luận Tự Động

        Cell này tạo một summary JSON để tiện đưa vào báo cáo.
        """
    ),
    code(
        """
        summary = {
            "created_at": pd.Timestamp.now().isoformat(),
            "feature_artifact": str(PROCESSED_DIR / "baseline_features.npz"),
            "feature_dim": int(X_raw.shape[1]),
            "n_samples": int(len(y)),
            "labels": LABEL_ORDER,
            "scenarios": [],
        }

        for row in best_rows.itertuples(index=False):
            scenario_name = row.scenario
            model_name = row.model
            val_row = metrics_df[
                (metrics_df["scenario"] == scenario_name)
                & (metrics_df["model"] == model_name)
                & (metrics_df["split"] == "validation")
            ].iloc[0]
            test_row = metrics_df[
                (metrics_df["scenario"] == scenario_name)
                & (metrics_df["model"] == model_name)
                & (metrics_df["split"] == "test")
            ].iloc[0]
            summary["scenarios"].append({
                "scenario": scenario_name,
                "best_model_selected_by_validation": model_name,
                "validation": {
                    "accuracy": float(val_row["accuracy"]),
                    "macro_f1": float(val_row["macro_f1"]),
                    "weighted_f1": float(val_row["weighted_f1"]),
                },
                "test": {
                    "accuracy": float(test_row["accuracy"]),
                    "macro_f1": float(test_row["macro_f1"]),
                    "weighted_f1": float(test_row["weighted_f1"]),
                },
                "best_params": json.loads(val_row["best_params"]) if isinstance(val_row["best_params"], str) and val_row["best_params"].startswith("{") else {},
            })

        (REPORT_DIR / "strong_classical_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        """
    ),
    md(
        """
        ## 11. Đóng Gói Kết Quả

        Kaggle sẽ tạo `strong_classical_baseline_results.zip` trong working directory để tải về.
        """
    ),
    code(
        """
        package_base = NOTEBOOK_DIR / "strong_classical_baseline_results"
        zip_path = shutil.make_archive(str(package_base), "zip", root_dir=OUTPUT_DIR)

        rar_path = NOTEBOOK_DIR / "strong_classical_baseline_results.rar"
        rar_exe = shutil.which("rar")
        if rar_exe:
            if rar_path.exists():
                rar_path.unlink()
            os.system(f'"{rar_exe}" a -r "{rar_path}" "{OUTPUT_DIR}"')
            print("RAR package:", rar_path)
        else:
            print("RAR executable not found. ZIP package is ready instead.")

        print("ZIP package:", zip_path)
        """
    ),
    md(
        """
        ## 12. Ghi Chú Đưa Vào Báo Cáo

        - Kết quả strict split là kết quả chính vì tránh đánh giá quá dễ theo speaker/recording style.
        - Kết quả random split dùng để so với nhiều paper/Kaggle baseline, nhưng không nên xem là bằng chứng triển khai thực tế.
        - Nếu random split cao hơn strict split nhiều, điều đó cho thấy bài toán bị ảnh hưởng mạnh bởi speaker/domain shift.
        - Sau notebook này mới nên chuyển sang log-Mel CNN hoặc SSL embedding để chứng minh mô hình học biểu diễn tốt hơn hand-crafted features.
        """
    ),
]


nbf.write(nb, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH}")
