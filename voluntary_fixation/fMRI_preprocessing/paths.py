"""データ配置の集中管理。

パスはすべてここで絶対パスとして定義するので、各スクリプトは CWD に依存せず動く。
環境を移したときは、下の REPO_ROOT（= data/ と datasets_out/ の親）だけ直せばよい。
"""
import os

# ── ここだけ環境に合わせて直す ────────────────────────────────
# data/ と datasets_out/ を含むリポジトリのルート
REPO_ROOT = '/home/kazu/shared_response'
# ──────────────────────────────────────────────────────────────

# 手順0: fMRIPrep の出力ルート（この下に studyforrest-data-all_out/fmriprep-latest/fmriprep/ が並ぶ）
DATASETS_OUT = os.path.join(REPO_ROOT, 'datasets_out')

# studyforrest 派生データのルート
STUDYFORREST = os.path.join(REPO_ROOT, 'data', 'forrestgump', 'studyforrest')

BOLD_DIR     = os.path.join(STUDYFORREST, 'bold')      # 手順1 出力 / 手順2 入力（denoise 済み BOLD）
BOLD_INT_DIR = os.path.join(STUDYFORREST, 'bold_int')  # 手順2 の中間物（base.bin / *_all.bin / *_rois.bin）


def npz_dir(n_roi, side):
    """SRM 入力 npz の出力先。side='' → roi22_np/ , side='LR' → roi22LR_np/"""
    return os.path.join(STUDYFORREST, f'roi{n_roi}{side}_np')
