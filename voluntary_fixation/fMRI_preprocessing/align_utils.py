import datetime
import os
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize, LogNorm
# from scipy.ndimage.measurements import label
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from nltools.mask import collapse_mask
from nltools.data import Brain_Data, Adjacency
from nltools.stats import align
import json

matplotlib.use('Agg')


def obtain_mask(mask_x, rois, mask_id='', viz=False):
    if len(rois) == 1:
        roi_mask = mask_x[rois[0]]
        # print(f'# of vs in {mask_id}: {np.sum(roi_mask.data)}')
    else:
        ms = []
        for roi in rois:
            m = mask_x[roi]
            # print(f'# of vs in {mask_id}: {np.sum(m.data)}')
            ms.append(m.data)
        mask = mask_x.copy()
        mask.data = np.stack(ms)
        roi_mask = collapse_mask(mask)
        roi_mask.data[roi_mask.data > 0] = 1  # for extract_roi

    if viz:
        roi_mask.plot()
        # plt.show()
        plt.savefig(f'align_mask_{mask_id}.png')
        plt.close()

    return roi_mask


def plot_dims(n_dims, prefix=''):
    n_dims = np.array(n_dims)
    d_mean = np.mean(n_dims)
    d_median = np.median(n_dims)
    d_min = np.min(n_dims)
    d_max = np.max(n_dims)

    # TODO: normalize the height of annotation based on hist
    f, a = plt.subplots(figsize=(6,4))
    plt.hist(n_dims, density=True, histtype='step', color='k')
    plt.axvline(x=d_mean, linestyle='--', color='r')
    plt.text(d_mean + 10, 0.6 * 0.0001, f"mean: {d_mean:.1f}")
    plt.axvline(x=d_median, linestyle='--', color='g')
    plt.text(d_median + 10, 1.1 * 0.0001, f"median: {d_median}")
    plt.axvline(x=d_min, linestyle='--', color='b')
    plt.text(d_min + 10, 0.1 * 0.0001, f"min: {d_min}")
    plt.axvline(x=d_max, linestyle='--', color='b')
    plt.text(d_max + 10, 0.1 * 0.0001, f"max: {d_max}")
    plt.ylabel('Frequency')
    plt.xlabel('# of dimensions')
    plt.title('Distribution of # of dimensions')

    plt.savefig(prefix + 'dims.png')
    plt.close()


def viz_ha(hyperalign, prefix=''):
    f,a = plt.subplots(figsize=(15,5))
    plt.hist(hyperalign['isc'].values())
    isc_mean = np.mean(list(hyperalign['isc'].values()))
    plt.axvline(x=isc_mean, linestyle='--', color='red', linewidth=2)
    plt.ylabel('Frequency', fontsize=16)
    plt.xlabel('Voxel IRC Values', fontsize=16)
    plt.title(f'Hyperalignment IRC({isc_mean:.2f})', fontsize=18)
    plt.xlim([0, 1])
    # plt.show()
    plt.savefig(prefix + 'ha_irc.png')
    plt.close()
    print(f"Mean IRC: {isc_mean:.2}")


def viz_srm(srm, prefix=''):
    f,a = plt.subplots(figsize=(15, 5))
    plt.hist(srm['isc'].values())
    isc_mean = np.mean(list(srm['isc'].values()))
    plt.axvline(x=isc_mean, linestyle='--', color='red', linewidth=2)
    plt.ylabel('Frequency', fontsize=16)
    plt.xlabel('Voxel IRC Values', fontsize=16)
    plt.title(f'Shared Response Model IRC({isc_mean:.2f})', fontsize=18)
    plt.xlim([0, 1])
    # plt.show()
    plt.savefig(prefix + 'srm_irc.png')
    plt.close()
    print(f"Mean IRC: {isc_mean:.2}")


def compare_rms(train_data_vs_std, train_data_as_std, test_data_vs_std, test_data_as_std, rms_v_train, rms_a_train, rms_v_test, rms_a_test, evr_vs, evr_as, model, cond_prefix):
    left = np.array([1, 2])
    colorlist = ['r', 'w']
    edgecolorlist = ['k', 'k']

    f, a = plt.subplots(nrows=2, ncols=2, figsize=(6, 6), sharey=True)

    label = ['ha rms', 'data std']
    a[0,0].bar(left, [rms_v_train, train_data_vs_std], tick_label=label, color=colorlist, edgecolor=edgecolorlist, align='center')
    a[0,0].set_title(f'train(visual): {evr_vs:.2f}')

    label = ['ha rms', 'data std']
    a[0,1].bar(left, [rms_a_train, train_data_as_std], tick_label=label, color=colorlist, edgecolor=edgecolorlist, align='center')
    a[0,1].set_title(f'train(auditory): {evr_as:.2f}')

    label = ['ha rms', 'data std']
    a[1,0].bar(left, [rms_v_test, test_data_vs_std], tick_label=label, color=colorlist, edgecolor=edgecolorlist, align='center')
    a[1,0].set_title('test(visual)')

    label = ['ha rms', 'data std']
    a[1,1].bar(left, [rms_a_test, test_data_as_std], tick_label=label, color=colorlist, edgecolor=edgecolorlist, align='center')
    a[1,1].set_title('test(auditory)')

    plt.tight_layout()
    # plt.show()
    plt.savefig(cond_prefix + f'{model["name"]}_rms_summary.png')
    plt.close()


def compare_irc(isc_train_raw, isc_test_raw_learnt, isc_test_raw, isc_train, isc_test, model, cond_prefix):
    isc_train_raw_mean = np.mean(list(isc_train_raw.values()))
    isc_test_raw_learnt_mean = np.mean(list(isc_test_raw_learnt.values()))
    isc_test_raw_mean = np.mean(list(isc_test_raw.values()))
    isc_train_mean = np.mean(list(isc_train.values()))
    isc_test_mean = np.mean(list(isc_test.values()))

    labels = ['before', 'after', 'before(recalc)']
    bins = np.linspace(-1, 1, 20)
    f, a = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharey=True)

    a[0].hist(list(isc_train_raw.values()), bins, histtype='stepfilled', density=True, color='b', alpha=0.5, label=labels[0])
    a[0].hist(list(isc_train.values()), bins, histtype='stepfilled', density=True, color='r', alpha=0.5, label=labels[1])
    a[0].axvline(x=isc_train_raw_mean, linestyle='--', color='b', linewidth=2)
    a[0].axvline(x=isc_train_mean, linestyle='--', color='r', linewidth=2)
    a[0].set_ylabel('Frequency')
    a[0].set_xlabel('Voxel IRC Values')
    a[0].set_xlim([-1, 1])
    a[0].set_title(f'train: {isc_train_raw_mean:.2f} -> {isc_train_mean:.2f}')
    a[0].legend()

    a[1].hist(list(isc_test_raw_learnt.values()), bins, histtype='stepfilled', density=True, color='b', alpha=0.5, label=labels[0])
    a[1].hist(list(isc_test.values()), bins, histtype='stepfilled', density=True, color='r', alpha=0.5, label=labels[1])
    a[1].hist(list(isc_test_raw.values()), bins, histtype='step', density=True, color='g', alpha=0.5, label=labels[2])
    a[1].axvline(x=isc_test_raw_learnt_mean, linestyle='--', color='b', linewidth=2)
    a[1].axvline(x=isc_test_mean, linestyle='--', color='r', linewidth=2)
    a[1].axvline(x=isc_test_raw_mean, linestyle='--', color='g', linewidth=2)
    a[1].set_ylabel('Frequency')
    a[1].set_xlabel('Voxel IRC Values')
    a[1].set_xlim([-1, 1])
    a[1].set_title(f'test: {isc_test_raw_learnt_mean:.2f}({isc_test_raw_mean:.2f}) -> {isc_test_mean:.2f}')
    a[1].legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(cond_prefix + f'{model["name"]}_irc_summary.png')
    plt.close()


def compare_irc_timeseries(data, context=None, n_dim_show=5, n_sample_show=100):
    data_0 = data[0]
    data_1 = data[1]
    n_dim = data_0.shape[1]
    if n_dim_show > n_dim:
        n_dim_show = n_dim
    dims = np.random.choice(n_dim, size=n_dim_show, replace=False)
    n_sample = data_0.shape[0]
    if n_sample_show > n_sample:
        n_sample_show = n_sample


    now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    if context is not None:
        title = 'irc_ts_' + context + '_' + now
    else:
        title = 'irc_ts_' + now
    fig, ax = plt.subplots(n_dim_show, 1, figsize=(10, 10))
    # random selection of dimension to show?
    # show IRC value?
    for d, dim in enumerate(dims):
        ax[d].plot(data_0[:n_sample_show, dim], color='r')
        ax[d].plot(data_1[:n_sample_show, dim], color='b')

    plt.tight_layout()
    plt.savefig(title + '.png')


def calc_irc(data, n_features, context=None, viz=False):
    # Calculate Intersubject correlation on aligned components
    a = Adjacency()
    if isinstance(data[0], Brain_Data):
        data = [bd.data for bd in data]

    if viz:
        # visualize timeseries
        compare_irc_timeseries(data, context=context)

    # print("d", len(data), data[0].shape, data[1].shape)  # list of numpy array w/ (sample, feature)
    # print(np.array([x[:,0].T for x in data]).shape)
    # TODO: deal with zero padding
    for f in range(n_features):
        p_d = pairwise_distances(np.array([x[:, f].T for x in data]), metric="correlation",)
        p_s = np.arctanh(1 - p_d)  # Fisher's z transform
        a_f = Adjacency(p_s, metric="similarity",)
        # print(f, p_d, a_f)
        a = a.append(a_f)
    # print("a", a)
    # print(a.mean(axis=1))
    if n_features > 1:
        isc = dict(zip(np.arange(n_features), a.mean(axis=1)))
    else:
        isc = {0: a.mean(axis=1)}
    isc_mean = np.mean(list(isc.values()))
    # print("i", isc)
    return isc, isc_mean


def calc_irc_raw(data_raw, n_features, method='rank', learnt=None, context=None, viz=False):
    if isinstance(data_raw[0], Brain_Data):
        data_raw = [bd.data for bd in data_raw]
    if method == 'rank':
        # (Haxby et al., 2011)
        if learnt is None:
            # print("before corr")
            R = np.corrcoef(data_raw[0], data_raw[1], rowvar=False) # n_sample x n_dim -> 2n_dim x 2n_dim
            # print("after corr")
            n_voxel = int(R.shape[0] / 2)
            # option 1. find the top n_feature voxel in
            # print(R)
            R_0 = np.nan_to_num(R[:n_voxel,n_voxel:], nan=-1)
            R_1 = np.nan_to_num(R[n_voxel:,:n_voxel], nan=-1)
            # print(R_0, R_1)
            R_0_score = np.amax(R_0, axis=1) # 各行でmaxの相関値を抽出
            R_1_score = np.amax(R_1, axis=1)
            # print(R_0_score.shape, R_1_score.shape)
            # 大きい相関を持っていたVOXELをn_feature(500)個抽出
            ranked_idx_0 = np.argsort(R_0_score)[::-1][:n_features]
            ranked_idx_1 = np.argsort(R_1_score)[::-1][:n_features]
            # Q. find the best ordering of index?
            # print(ranked_idx_0, ranked_idx_1)
            # print(R_0_score[ranked_idx_0], R_1_score[ranked_idx_1])
            learnt = (ranked_idx_0, ranked_idx_1)
            # option 2. find the voxel combination(s) with high correlation
        else:
            ranked_idx_0, ranked_idx_1 = learnt
        data_reduced_0 = data_raw[0][:,ranked_idx_0]
        data_reduced_1 = data_raw[1][:,ranked_idx_1]
        data_reduced = [data_reduced_0, data_reduced_1]
    elif method == 'pca':
        if learnt is None:
            pca_0 = PCA(n_components=n_features)
            pca_1 = PCA(n_components=n_features)
            data_reduced_0 = pca_0.fit_transform(data_raw[0])
            data_reduced_1 = pca_1.fit_transform(data_raw[1])
            learnt = (pca_0, pca_1)
        else:
            pca_0, pca_1 = learnt
            data_reduced_0 = pca_0.transform(data_raw[0])
            data_reduced_1 = pca_1.transform(data_raw[1])
        data_reduced = [data_reduced_0, data_reduced_1]
    else:
        raise NotImplementedError()
    if context is not None:
        context = "raw_" + context
    else:
        context = "raw"
    isc, isc_mean = calc_irc(data_reduced, n_features, context=context, viz=viz)
    return isc, isc_mean, learnt


# https://qiita.com/ynakayama/items/7dc01f45caf6d87a981b
def draw_heatmap(data, row_labels, column_labels, figtitle="test"):
    fig, ax = plt.subplots(figsize=(24, 24))

    heatmap = ax.pcolor(data, norm=Normalize(vmin=-1, vmax=1), cmap="coolwarm")  # cmap="viridis"
    # heatmap = ax.pcolor(data, norm=LogNorm(vmin=-1, vmax=1))
    heatmap.set_clim(-1, 1)
    pp = fig.colorbar(heatmap, ax=ax, orientation="vertical")
    pp.set_label("IRC mean")

    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_yticklabels(row_labels, minor=False)
    ax.set_xticklabels(column_labels, minor=False)
    ax.set_title(figtitle)

    # plt.show()
    plt.savefig(figtitle + ".png")
    plt.close()


def draw_heatmap_detailed(data, labels, figtitle="test"):
    # eg. draw_heatmap_detailed(np.random.rand(180,180), [str(i + 1)*20 for i in range(180)])
    n_label = len(labels)
    assert n_label == data.shape[0] == data.shape[1]
    # fig, ax = plt.subplots(1, 3, figsize=(72, 24))
    fig = plt.figure(figsize=(n_label, int(n_label/3)))
    label_width = max(1, n_label - 2)
    spec = gridspec.GridSpec(ncols=4, nrows=1, width_ratios=[n_label,n_label,n_label-label_width,label_width])
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])
    ax2 = fig.add_subplot(spec[2])
    ax3 = fig.add_subplot(spec[3])

    # sns.heatmap(data, square=True, vmin=-1, vmax=1, cmap='')  # 'RdBu_r'
    heatmap0 = ax0.pcolor(data, norm=Normalize(vmin=-1, vmax=1), cmap="coolwarm")
    # heatmap0 = ax[0].pcolor(data, norm=LogNorm(vmin=-1, vmax=1))
    heatmap0.set_clim(-1, 1)
    pp0 = fig.colorbar(heatmap0, ax=ax0, orientation="vertical")
    pp0.set_label("IRC mean")
    ax0.set_xticks(np.arange(n_label) + 0.5, minor=False)
    ax0.set_yticks(np.arange(n_label) + 0.5, minor=False)
    ax0.invert_yaxis()
    ax0.xaxis.tick_top()
    ax0.set_xticklabels(range(n_label), minor=False)
    ax0.set_yticklabels(range(n_label), minor=False)
    ax0.set_title(figtitle + " (Normalized)")

    heatmap1 = ax1.pcolor(data, cmap="viridis")
    pp1 = fig.colorbar(heatmap1, ax=ax1, orientation="vertical")
    pp1.set_label("IRC mean")
    ax1.set_xticks(np.arange(n_label) + 0.5, minor=False)
    ax1.set_yticks(np.arange(n_label) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    ax1.set_xticklabels(range(n_label), minor=False)
    ax1.set_yticklabels(range(n_label), minor=False)
    ax1.set_title("(Not Normalized)")

    # use mean instead of sum?
    # ignore diagonal when taking sum?
    data_sum = np.sum(data, axis=1, keepdims=True)
    data_max_idx = np.argmax(data_sum)  # get the first index(use top-k?)
    data_sort_idx = np.argsort(data_sum.reshape(-1))[::-1]
    data_sum_sorted = data_sum[data_sort_idx,:]
    labels_sorted = [labels[dsi] for dsi in data_sort_idx]
    # print(data_sum.shape, data_max_idx.shape, data_sort_idx.shape, data_sum_sorted.shape)
    heatmap2 = ax2.pcolor(data_sum_sorted, cmap="viridis")
    ax2.set_yticks(np.arange(n_label) + 0.5, minor=False)
    ax2.set_xticks([])
    ax2.invert_yaxis()
    ax2.yaxis.tick_right()
    ax2.set_yticklabels(labels_sorted, minor=False)
    ax2.set_xticklabels([])
    # ax2.set_title(f"(Sum): max at {data_max_idx}, {labels[data_max_idx]}")
    ax2.set_title("(Sorted Sum)")

    ax3.axis('off')

    # plt.show()
    # plt.tight_layout()
    plt.savefig(figtitle + "_detailed.png")
    plt.close()


def remove_existing_file(fn):
    if os.path.exists(fn):
        print(f"Removing: {fn}")
        os.remove(fn)


def load_dumps(f):
    obj = []
    while True:
        try:
            obj.append(pickle.load(f))
        except:
            break
    return obj


def load_ircs(irc_fn, n_roi):
    with open(irc_fn, 'rb') as f:
        irc_obj = load_dumps(f)

    irc_means = np.zeros([n_roi, n_roi], dtype=float)
    # print(results)
    # pack data
    # for comb, (roi_1, roi_2) in enumerate(roi_combs):
    for irc_comb in irc_obj:
        roi_1, roi_2, irc = irc_comb[0], irc_comb[1], irc_comb[2]
        '''
        if roi_1 == roi_2:
            continue
        if roi_1 < roi_2:
            # irc_means[roi_1, roi_2] = results[comb][0]
            irc_means[roi_1, roi_2] = irc
        else:
            irc_means[roi_1, roi_2] = irc_means[roi_2, roi_1]
        '''
        if roi_1 < roi_2:
            irc_means[roi_1, roi_2] = irc
    for irc_comb in irc_obj:
        roi_1, roi_2, irc = irc_comb[0], irc_comb[1], irc_comb[2]
        if roi_1 > roi_2:
            irc_means[roi_1, roi_2] = irc_means[roi_2, roi_1]

    return irc_means

def my_align_2_rois(cond_str:str, int_base:str, train_comb:tuple,test_comb:tuple,
                    n_latent:int=-1, n_feature:int=200):
    train_roi_1_, train_roi_2_ = train_comb
    test_roi_1_, test_roi_2_ = test_comb
    del train_comb, test_comb

    n_train = train_roi_1_.shape[0] # n_electrode数？ではなくサンプル数っぽい（けど、上のとり方と合わない気もする）
    roi_1_dim = train_roi_1_.shape[1] # n_electrodes or n_dims of features
    roi_2_dim = train_roi_2_.shape[1]
    n_test = test_roi_1_.shape[0]
    # print(n_train, n_test, roi_1_dim, roi_2_dim)
    # zero padding
    if n_latent <= 0:
        if roi_1_dim > roi_2_dim:
            n_dim_l = roi_2_dim
            n_dim_h = roi_1_dim
            n_pad_dim = roi_1_dim - roi_2_dim
            # roi1の次元とroi2の次元の差分をpaddingすることで大きさを揃える
            train_roi_2_ = np.concatenate([train_roi_2_.data, np.zeros([n_train, n_pad_dim])], axis=1)
            test_roi_2_ = np.concatenate([test_roi_2_.data, np.zeros([n_test, n_pad_dim])], axis=1)
        elif roi_1_dim < roi_2_dim:
            n_dim_l = roi_1_dim
            n_dim_h = roi_2_dim
            n_pad_dim = roi_2_dim - roi_1_dim
            train_roi_1_ = np.concatenate([train_roi_1_.data, np.zeros([n_train, n_pad_dim])], axis=1)
            test_roi_1_ = np.concatenate([test_roi_1_.data, np.zeros([n_test, n_pad_dim])], axis=1)
        else:
            n_dim_l = roi_1_dim
            n_dim_h = roi_1_dim

        if n_feature > n_dim_h:
            n_pad_dim = n_feature - n_dim_h
            train_roi_1_ = np.concatenate([train_roi_1_.data, np.zeros([n_train, n_pad_dim])], axis=1)
            train_roi_2_ = np.concatenate([train_roi_2_.data, np.zeros([n_train, n_pad_dim])], axis=1)
            test_roi_1_ = np.concatenate([test_roi_1_.data, np.zeros([n_test, n_pad_dim])], axis=1)
            test_roi_2_ = np.concatenate([test_roi_2_.data, np.zeros([n_test, n_pad_dim])], axis=1)
            n_dim_h = n_feature
    else:
        n_dim_l = n_latent
        n_dim_h = n_latent
        ## done: n_latent > min(roi_1_dim, roi_2_dim)
    # import pdb; pdb.set_trace()
    train_data = [train_roi_1_, train_roi_2_] # n_train_samples x n_dim_h
    test_data = [test_roi_1_, test_roi_2_] # n_test_samples x n_dim_h

    # print(n_dim)
    # print(train_data[0].shape()[0], train_data[0].shape()[1])
    # print(train_data[1].shape()[0], train_data[1].shape()[1])
    isc_raw_method = 'rank'
    # isc_raw_method = 'pca'
    # print(f"Original ISC(train): {isc_train_raw_mean}")
    # print(f"Original ISC(test): {isc_test_raw_mean}")

    viz = False
    if n_feature > 0: # n_featureが負の場合, n_dim_hを設定する ->これだとircの計算機に 0_paddingの0も含めれてしまう
        # train linear map
        isc_train_raw, isc_train_raw_mean, learnt = calc_irc_raw(train_data, n_feature, method=isc_raw_method, context='train', viz=viz)
        # predict on test data
        isc_test_raw_learnt, isc_test_raw_learnt_mean, _ = calc_irc_raw(test_data, n_feature, method=isc_raw_method, learnt=learnt, context='test_learnt', viz=viz)
        # predict on test data ? (not learnt = random weight?)
        isc_test_raw, isc_test_raw_mean, _ = calc_irc_raw(test_data, n_feature, method=isc_raw_method, context='test', viz=viz)
    else:
        isc_train_raw, isc_train_raw_mean, learnt = calc_irc_raw(train_data, n_dim_l, method=isc_raw_method, context='train', viz=viz)  # n_dim_h?
        isc_test_raw_learnt, isc_test_raw_learnt_mean, _ = calc_irc_raw(test_data, n_dim_l, method=isc_raw_method, learnt=learnt, context='test_learnt', viz=viz)  # n_dim_h?
        isc_test_raw, isc_test_raw_mean, _ = calc_irc_raw(test_data, n_dim_l, method=isc_raw_method, context='test', viz=viz)  # n_dim_h?

    # model_type = 'ha'
    model_type = 'srm_p'
    # model_type = 'srm_d'
    if model_type == 'ha':
        hyperalign = align(train_data, method='procrustes')
        # model = {"method": "procrustes", "alignment": hyperalign, "name": "ha", "n_feature": n_dim_l}
        model = {"method": "procrustes", "alignment": hyperalign, "name": "ha", "n_feature": n_dim_h}
    elif (model_type == 'srm_p') and (n_feature > 0):
        srm_p = align(train_data, method='probabilistic_srm', n_features=n_feature)
        model = {"method": "probabilistic_srm", "alignment": srm_p, "name": "srm_p", "n_feature": n_feature}
    elif (model_type == 'srm_p') and (n_feature <= 0):
        # TODO: cope with correlation of zero vector to have n_features and n_feature the same (?), dim_h instead of dim_l ?
        srm_p = align(train_data, method='probabilistic_srm', n_features=n_dim_h)
        # model = {"method": "probabilistic_srm", "alignment": srm_p, "name": "srm_p", "n_feature": n_dim_l}
        model = {"method": "probabilistic_srm", "alignment": srm_p, "name": "srm_p", "n_feature": n_dim_h}
    elif (model_type == 'srm_d') and (n_feature > 0):
        srm_d = align(train_data, method='deterministic_srm', n_features=n_feature)
        model = {"method": "deterministic_srm", "alignment": srm_d, "name": "srm_d", "n_feature": n_feature}
    elif (model_type == 'srm_d') and (n_feature <= 0):
        srm_d = align(train_data, method='deterministic_srm', n_features=n_dim_h)
        # model = {"method": "deterministic_srm", "alignment": srm_d, "name": "srm_d", "n_feature": n_dim_l}
        model = {"method": "deterministic_srm", "alignment": srm_d, "name": "srm_d", "n_feature": n_dim_h}
    # print(model["name"])

    isc_train, isc_train_mean = calc_irc(model["alignment"]["transformed"], model["n_feature"], viz=viz)  # model["alignment"]["isc"]
    # print(f"Validate ISC(train): {isc_train_mean}")

    roi_1_p = model["alignment"]['transformation_matrix'][0] # .T # 直交行列の制約を入れているので、転置で逆行列になる
    roi_2_p = model["alignment"]['transformation_matrix'][1] # .T
    # import pdb; pdb.set_trace()
    transformed_roi_1 = np.dot(test_data[0], roi_1_p)
    transformed_roi_2 = np.dot(test_data[1], roi_2_p)
    transformed = [transformed_roi_1, transformed_roi_2]
    isc_test, isc_test_mean = calc_irc(transformed, model["n_feature"], viz=viz)
    # print(f"Validate ISC(test): {isc_test_mean}")

    # visualize
    # compare_isc(isc_train_raw, isc_test_raw_learnt, isc_test_raw, isc_train, isc_test, model, cond_prefix)

    # float(isc_train_raw_mean), float(isc_test_raw_learnt_mean), float(isc_test_raw_mean), float(isc_train_mean), float(isc_test_mean)
    ret = {'isc_train_raw_mean': float(isc_train_raw_mean),
           'isc_test_raw_learnt_mean': float(isc_test_raw_learnt_mean),
           'isc_test_raw_mean': float(isc_test_raw_mean),
           'isc_train_mean': float(isc_train_mean),
           'isc_test_mean': float(isc_test_mean)}
    savefile = os.path.join(int_base, cond_str+'/', "isc.json")
    os.makedirs(os.path.dirname(savefile), exist_ok=True)
    with open(savefile, 'w') as f:
        json.dump(ret, f)
    print('save inter-subject disimilarity score to: ', savefile)
    return model, train_data, test_data




def align_2_rois(cond_str, int_base, result_base, roi_comb):
    roi_1, roi_2 = roi_comb
    if roi_1 >= roi_2:
        # return 0, 0, 0, 0
        # return 0
        return

    params_fn = result_base + cond_str + 'params.npz'
    params = np.load(params_fn, allow_pickle=True)  # be careful
    n_latent = int(params['n_latent'])
    n_feature = int(params['n_feature'])
    # n_feature = 20  # override
    cond_str_ = "_".join(cond_str.split("_")[:-2]) + "_"
    train_fn = int_base + cond_str_ + 'train.bin'
    test_fn = int_base + cond_str_ + 'test.bin'
    # test_fn = train_fn  # override
    irc_train_raw_fn = int_base + cond_str + "irc_train_raw.bin"
    irc_test_raw_learnt_fn = int_base + cond_str + "irc_test_raw_learnt.bin"
    irc_test_raw_fn = int_base + cond_str + "irc_test_raw.bin"
    irc_train_fn = int_base + cond_str + "irc_train.bin"
    irc_test_fn = int_base + cond_str + "irc_test.bin"

    with open(train_fn, 'rb') as p:
        train_data_rois = pickle.load(p)
    with open(test_fn, 'rb') as p:
        test_data_rois = pickle.load(p)

    cond_prefix = cond_str + f"ROI-{roi_1}_ROI-{roi_2}_"
    # print(cond_prefix)

    train_roi_1_ = train_data_rois[roi_1].copy()
    train_roi_2_ = train_data_rois[roi_2].copy()
    test_roi_1_ = test_data_rois[roi_1].copy()
    test_roi_2_ = test_data_rois[roi_2].copy()
    del train_data_rois, test_data_rois

    n_train = train_roi_1_.shape()[0]
    roi_1_dim = train_roi_1_.shape()[1]
    roi_2_dim = train_roi_2_.shape()[1]
    n_test = test_roi_1_.shape()[0]
    # print(n_train, n_test, roi_1_dim, roi_2_dim)
    # zero padding
    if n_latent <= 0:
        if roi_1_dim > roi_2_dim:
            n_dim_l = roi_2_dim
            n_dim_h = roi_1_dim
            n_pad_dim = roi_1_dim - roi_2_dim
            train_roi_2_.data = np.concatenate([train_roi_2_.data, np.zeros([n_train, n_pad_dim])], axis=1)
            test_roi_2_.data = np.concatenate([test_roi_2_.data, np.zeros([n_test, n_pad_dim])], axis=1)
        elif roi_1_dim < roi_2_dim:
            n_dim_l = roi_1_dim
            n_dim_h = roi_2_dim
            n_pad_dim = roi_2_dim - roi_1_dim
            train_roi_1_.data = np.concatenate([train_roi_1_.data, np.zeros([n_train, n_pad_dim])], axis=1)
            test_roi_1_.data = np.concatenate([test_roi_1_.data, np.zeros([n_test, n_pad_dim])], axis=1)
        else:
            n_dim_l = roi_1_dim
            n_dim_h = roi_1_dim

        if n_feature > n_dim_h:
            n_pad_dim = n_feature - n_dim_h
            train_roi_1_.data = np.concatenate([train_roi_1_.data, np.zeros([n_train, n_pad_dim])], axis=1)
            train_roi_2_.data = np.concatenate([train_roi_2_.data, np.zeros([n_train, n_pad_dim])], axis=1)
            test_roi_1_.data = np.concatenate([test_roi_1_.data, np.zeros([n_test, n_pad_dim])], axis=1)
            test_roi_2_.data = np.concatenate([test_roi_2_.data, np.zeros([n_test, n_pad_dim])], axis=1)
            n_dim_h = n_feature
    else:
        n_dim_l = n_latent
        n_dim_h = n_latent
        ## done: n_latent > min(roi_1_dim, roi_2_dim)

    train_data = [train_roi_1_, train_roi_2_]
    test_data = [test_roi_1_, test_roi_2_]

    # print(n_dim)
    # print(train_data[0].shape()[0], train_data[0].shape()[1])
    # print(train_data[1].shape()[0], train_data[1].shape()[1])
    isc_raw_method = 'rank'
    # isc_raw_method = 'pca'
    # print(f"Original ISC(train): {isc_train_raw_mean}")
    # print(f"Original ISC(test): {isc_test_raw_mean}")

    viz = False
    if n_feature > 0:
        isc_train_raw, isc_train_raw_mean, learnt = calc_irc_raw(train_data, n_feature, method=isc_raw_method, context='train', viz=viz)
        isc_test_raw_learnt, isc_test_raw_learnt_mean, _ = calc_irc_raw(test_data, n_feature, method=isc_raw_method, learnt=learnt, context='test_learnt', viz=viz)
        isc_test_raw, isc_test_raw_mean, _ = calc_irc_raw(test_data, n_feature, method=isc_raw_method, context='test', viz=viz)
    else:
        isc_train_raw, isc_train_raw_mean, learnt = calc_irc_raw(train_data, n_dim_l, method=isc_raw_method, context='train', viz=viz)  # n_dim_h?
        isc_test_raw_learnt, isc_test_raw_learnt_mean, _ = calc_irc_raw(test_data, n_dim_l, method=isc_raw_method, learnt=learnt, context='test_learnt', viz=viz)  # n_dim_h?
        isc_test_raw, isc_test_raw_mean, _ = calc_irc_raw(test_data, n_dim_l, method=isc_raw_method, context='test', viz=viz)  # n_dim_h?

    # model_type = 'ha'
    model_type = 'srm_p'
    # model_type = 'srm_d'

    if model_type == 'ha':
        hyperalign = align(train_data, method='procrustes')
        # model = {"method": "procrustes", "alignment": hyperalign, "name": "ha", "n_feature": n_dim_l}
        model = {"method": "procrustes", "alignment": hyperalign, "name": "ha", "n_feature": n_dim_h}
    elif (model_type == 'srm_p') and (n_feature > 0):
        srm_p = align(train_data, method='probabilistic_srm', n_features=n_feature)
        model = {"method": "probabilistic_srm", "alignment": srm_p, "name": "srm_p", "n_feature": n_feature}
    elif (model_type == 'srm_p') and (n_feature <= 0):
        # TODO: cope with correlation of zero vector to have n_features and n_feature the same (?), dim_h instead of dim_l ?
        srm_p = align(train_data, method='probabilistic_srm', n_features=n_dim_h)
        # model = {"method": "probabilistic_srm", "alignment": srm_p, "name": "srm_p", "n_feature": n_dim_l}
        model = {"method": "probabilistic_srm", "alignment": srm_p, "name": "srm_p", "n_feature": n_dim_h}
    elif (model_type == 'srm_d') and (n_feature > 0):
        srm_d = align(train_data, method='deterministic_srm', n_features=n_feature)
        model = {"method": "deterministic_srm", "alignment": srm_d, "name": "srm_d", "n_feature": n_feature}
    elif (model_type == 'srm_d') and (n_feature <= 0):
        srm_d = align(train_data, method='deterministic_srm', n_features=n_dim_h)
        # model = {"method": "deterministic_srm", "alignment": srm_d, "name": "srm_d", "n_feature": n_dim_l}
        model = {"method": "deterministic_srm", "alignment": srm_d, "name": "srm_d", "n_feature": n_dim_h}
    # print(model["name"])

    isc_train, isc_train_mean = calc_irc(model["alignment"]["transformed"], model["n_feature"], viz=viz)  # model["alignment"]["isc"]
    # print(f"Validate ISC(train): {isc_train_mean}")

    roi_1_p = model["alignment"]['transformation_matrix'][0].data.T
    roi_2_p = model["alignment"]['transformation_matrix'][1].data.T
    transformed_roi_1 = np.dot(test_data[0].data, roi_1_p)
    transformed_roi_2 = np.dot(test_data[1].data, roi_2_p)
    transformed = [transformed_roi_1, transformed_roi_2]
    isc_test, isc_test_mean = calc_irc(transformed, model["n_feature"], viz=viz)
    # print(f"Validate ISC(test): {isc_test_mean}")

    # visualize
    # compare_isc(isc_train_raw, isc_test_raw_learnt, isc_test_raw, isc_train, isc_test, model, cond_prefix)

    # float(isc_train_raw_mean), float(isc_test_raw_learnt_mean), float(isc_test_raw_mean), float(isc_train_mean), float(isc_test_mean)
    with open(irc_train_raw_fn, 'ab') as f:
        pickle.dump([roi_1, roi_2, float(isc_train_raw_mean)], f)
    with open(irc_test_raw_learnt_fn, 'ab') as f:
        pickle.dump([roi_1, roi_2, float(isc_test_raw_learnt_mean)], f)
    with open(irc_test_raw_fn, 'ab') as f:
        pickle.dump([roi_1, roi_2, float(isc_test_raw_mean)], f)
    with open(irc_train_fn, 'ab') as f:
        pickle.dump([roi_1, roi_2, float(isc_train_mean)], f)
    with open(irc_test_fn, 'ab') as f:
        pickle.dump([roi_1, roi_2, float(isc_test_mean)], f)

    # return 0
    return


def load_srms(srm_fn, n_roi):
    with open(srm_fn, 'rb') as f:
        srm_obj = load_dumps(f)
    # print(srm_obj, type(srm_obj), len(srm_obj))

    srms = np.empty([n_roi, n_roi], dtype=object)
    # pack data
    for srm_comb in srm_obj:
        roi_1, roi_2, srm_1, srm_2 = srm_comb[0], srm_comb[1], srm_comb[2], srm_comb[3]
        # print(roi_1, roi_2, srm_1.shape, srm_2.shape)
        # if roi_1 >= roi_2:
        #     continue
        # if roi_1 < roi_2:
        assert roi_1 < roi_2
        srms[roi_1, roi_2] = srm_1
        srms[roi_2, roi_1] = srm_2

    return srms


def align_2_rois_ext(cond_str, int_base, result_base, roi_comb):
    roi_1, roi_2 = roi_comb
    if roi_1 >= roi_2:
        # return 0, 0, 0, 0
        return

    # ext_str = "_ext"
    ext_str = ""
    params_fn = result_base + cond_str + f'params{ext_str}.npz'
    params = np.load(params_fn, allow_pickle=True)  # be careful
    n_latent = int(params['n_latent'])
    n_feature = int(params['n_feature'])

    cond_str_1 = "_".join(cond_str.split("_")[:3]) + "_"
    data_rois_fn = int_base + cond_str_1 + f"rois{ext_str}.bin"
    with open(data_rois_fn, 'rb') as p:
        data_rois_ = pickle.load(p)
    roi_names, n_dims = data_rois_

    cond_str_2 = "_".join(cond_str.split("_")[:-2]) + "_"
    train_fn = int_base + cond_str_2 + f'train{ext_str}.bin'
    test_fn = int_base + cond_str_2 + f'test{ext_str}.bin'
    with open(train_fn, 'rb') as p:
        train_data_rois = pickle.load(p)
    with open(test_fn, 'rb') as p:
        test_data_rois = pickle.load(p)

    train_srm_fn = int_base + cond_str + f"srm_train{ext_str}.bin"
    # test_srm_fn = int_base + cond_str + f"srm_test{ext_str}.bin"
    # print(train_srm_fn)

    # cond_prefix = cond_str + f"discover_ROI-{roi_names[roi_1]}_ROI-{roi_names[roi_2]}_"
    # print(cond_prefix)

    train_roi_1_ = train_data_rois[roi_1].copy()
    train_roi_2_ = train_data_rois[roi_2].copy()
    test_roi_1_ = test_data_rois[roi_1].copy()
    test_roi_2_ = test_data_rois[roi_2].copy()
    del train_data_rois, test_data_rois

    n_train = train_roi_1_.shape()[0]
    roi_1_dim = train_roi_1_.shape()[1]
    roi_2_dim = train_roi_2_.shape()[1]
    n_test = test_roi_1_.shape()[0]
    # print(n_train, n_test, roi_1_dim, roi_2_dim)
    # zero padding
    if n_latent <= 0:
        if roi_1_dim > roi_2_dim:
            n_dim_l = roi_2_dim
            n_dim_h = roi_1_dim
            n_pad_dim = roi_1_dim - roi_2_dim
            train_roi_2_.data = np.concatenate([train_roi_2_.data, np.zeros([n_train, n_pad_dim])], axis=1)
            test_roi_2_.data = np.concatenate([test_roi_2_.data, np.zeros([n_test, n_pad_dim])], axis=1)
        elif roi_1_dim < roi_2_dim:
            n_dim_l = roi_1_dim
            n_dim_h = roi_2_dim
            n_pad_dim = roi_2_dim - roi_1_dim
            train_roi_1_.data = np.concatenate([train_roi_1_.data, np.zeros([n_train, n_pad_dim])], axis=1)
            test_roi_1_.data = np.concatenate([test_roi_1_.data, np.zeros([n_test, n_pad_dim])], axis=1)
        else:
            n_dim_l = roi_1_dim
            n_dim_h = roi_1_dim

        if n_feature > n_dim_h:
            n_pad_dim = n_feature - n_dim_h
            train_roi_1_.data = np.concatenate([train_roi_1_.data, np.zeros([n_train, n_pad_dim])], axis=1)
            train_roi_2_.data = np.concatenate([train_roi_2_.data, np.zeros([n_train, n_pad_dim])], axis=1)
            test_roi_1_.data = np.concatenate([test_roi_1_.data, np.zeros([n_test, n_pad_dim])], axis=1)
            test_roi_2_.data = np.concatenate([test_roi_2_.data, np.zeros([n_test, n_pad_dim])], axis=1)
            n_dim_h = n_feature
    else:
        n_dim_l = n_latent
        n_dim_h = n_latent
        ## done: n_latent > min(roi_1_dim, roi_2_dim)

    train_data = [train_roi_1_, train_roi_2_]
    test_data = [test_roi_1_, test_roi_2_]

    # print(n_dim)
    # print(train_data[0].shape()[0], train_data[0].shape()[1])
    # print(train_data[1].shape()[0], train_data[1].shape()[1])
    isc_raw_method = 'rank'
    # isc_raw_method = 'pca'
    # print(f"Original ISC(train): {isc_train_raw_mean}")
    # print(f"Original ISC(test): {isc_test_raw_mean}")

    viz = False
    if n_feature > 0:
        isc_train_raw, isc_train_raw_mean, learnt = calc_irc_raw(train_data, n_feature, method=isc_raw_method, context='train', viz=viz)
        isc_test_raw_learnt, isc_test_raw_learnt_mean, _ = calc_irc_raw(test_data, n_feature, method=isc_raw_method, learnt=learnt, context='test_learnt', viz=viz)
        isc_test_raw, isc_test_raw_mean, _ = calc_irc_raw(test_data, n_feature, method=isc_raw_method, context='test', viz=viz)
    else:
        isc_train_raw, isc_train_raw_mean, learnt = calc_irc_raw(train_data, n_dim_l, method=isc_raw_method, context='train', viz=viz)  # n_dim_h?
        isc_test_raw_learnt, isc_test_raw_learnt_mean, _ = calc_irc_raw(test_data, n_dim_l, method=isc_raw_method, learnt=learnt, context='test_learnt', viz=viz)  # n_dim_h?
        isc_test_raw, isc_test_raw_mean, _ = calc_irc_raw(test_data, n_dim_l, method=isc_raw_method, context='test', viz=viz)  # n_dim_h?

    # model_type = 'ha'
    model_type = 'srm_p'
    # model_type = 'srm_d'

    if model_type == 'ha':
        hyperalign = align(train_data, method='procrustes')
        # model = {"method": "procrustes", "alignment": hyperalign, "name": "ha", "n_feature": n_dim_l}
        model = {"method": "procrustes", "alignment": hyperalign, "name": "ha", "n_feature": n_dim_h}
    elif (model_type == 'srm_p') and (n_feature > 0):
        srm_p = align(train_data, method='probabilistic_srm', n_features=n_feature)
        model = {"method": "probabilistic_srm", "alignment": srm_p, "name": "srm_p", "n_feature": n_feature}
    elif (model_type == 'srm_p') and (n_feature <= 0):
        # TODO: cope with correlation of zero vector to have n_features and n_feature the same (?), dim_h instead of dim_l ?
        srm_p = align(train_data, method='probabilistic_srm', n_features=n_dim_h)
        # model = {"method": "probabilistic_srm", "alignment": srm_p, "name": "srm_p", "n_feature": n_dim_l}
        model = {"method": "probabilistic_srm", "alignment": srm_p, "name": "srm_p", "n_feature": n_dim_h}
    elif (model_type == 'srm_d') and (n_feature > 0):
        srm_d = align(train_data, method='deterministic_srm', n_features=n_feature)
        model = {"method": "deterministic_srm", "alignment": srm_d, "name": "srm_d", "n_feature": n_feature}
    elif (model_type == 'srm_d') and (n_feature <= 0):
        srm_d = align(train_data, method='deterministic_srm', n_features=n_dim_h)
        # model = {"method": "deterministic_srm", "alignment": srm_d, "name": "srm_d", "n_feature": n_dim_l}
        model = {"method": "deterministic_srm", "alignment": srm_d, "name": "srm_d", "n_feature": n_dim_h}
    # print(model["name"])

    isc_train, isc_train_mean = calc_irc(model["alignment"]["transformed"], model["n_feature"], context='train', viz=viz)  # model["alignment"]["isc"]
    # print(f"Validate ISC(train): {isc_train_mean}")

    roi_1_p = model["alignment"]['transformation_matrix'][0].data.T
    roi_2_p = model["alignment"]['transformation_matrix'][1].data.T
    transformed_roi_1 = np.dot(test_data[0].data, roi_1_p)
    transformed_roi_2 = np.dot(test_data[1].data, roi_2_p)
    transformed = [transformed_roi_1, transformed_roi_2]
    # transformed_mean = (transformed_roi_1 + transformed_roi_2) / 2  # does this substitute the common model?
    isc_test, isc_test_mean = calc_irc(transformed, model["n_feature"], context='test', viz=viz)
    # print(f"Validate ISC(test): {isc_test_mean}")

    iscs = {
        'train_raw': isc_train_raw,
        'test_raw_learnt': isc_test_raw_learnt,
        'test_raw': isc_test_raw,
        'train': isc_train,
        'test': isc_test
    }

    # train_transformed_mean = (model["alignment"]["transformed"][0] + model["alignment"]["transformed"][1]) / 2
    # test_transformed_mean = transformed_mean

    # Q. store each value instead of mean?
    # mean_fn = result_base + cond_str + f"mean_ext_{roi_1}_{roi_2}.npz"
    # np.savez(mean_fn, train=train_transformed_mean, test=test_transformed_mean)
    with open(train_srm_fn, 'ab') as f:
        pickle.dump([roi_1, roi_2, model["alignment"]["transformed"][0], model["alignment"]["transformed"][1]], f)
    # with open(test_srm_fn, 'ab') as f:
    #     pickle.dump([roi_1, roi_2, transformed_roi_1, transformed_roi_2], f)

    # model_fn = result_base + cond_str + "model_ext.bin"
    # test_mean_fn = result_base + cond_str + "test_mean_ext.bin"
    # iscs_fn = result_base + cond_str + "ircs_ext.npz"
    # with open(model_fn, 'wb') as p:
    #     pickle.dump(model, p)
    # with open(test_mean_fn, 'wb') as p:
    #     pickle.dump(transformed_mean, p)
    # with open(iscs_fn, 'wb') as p:
    #     pickle.dump(iscs, p)
    # np.savez(iscs_fn, **iscs)

    # visualize
    # compare_irc(isc_train_raw, isc_test_raw_learnt, isc_test_raw, isc_train, isc_test, model, cond_prefix)

    # return model, transformed_mean, iscs
    # return float(isc_train_raw_mean), float(isc_test_raw_learnt_mean), float(isc_test_raw_mean), float(isc_train_mean), float(isc_test_mean)
    return


def align_new_rois(cond_str, new_roi):
    model_fn = cond_str + "model_ext.bin"
    with open(model_fn, 'rb') as p:
        model = pickle.load(p)
    test_mean_fn = cond_str + "test_mean_ext.bin"
    with open(test_mean_fn, 'rb') as p:
        test_mean = pickle.load(p)

    params_fn = cond_str + 'params_ext.npz'
    params = np.load(params_fn, allow_pickle=True)  # be careful
    n_latent = int(params['n_latent'])
    # n_feature = int(params['n_feature'])
    cond_str_1 = "_".join(cond_str.split("_")[:2]) + "_"
    data_rois_fn = cond_str_1 + "rois_ext.bin"
    with open(data_rois_fn, 'rb') as p:
        data_rois_ = pickle.load(p)
    roi_names, n_dims = data_rois_
    cond_str_2 = "_".join(cond_str.split("_")[:-2]) + "_"
    train_fn = cond_str_2 + 'train_ext.bin'
    test_fn = cond_str_2 + 'test_ext.bin'
    with open(train_fn, 'rb') as p:
        train_data_rois = pickle.load(p)
    with open(test_fn, 'rb') as p:
        test_data_rois = pickle.load(p)

    cond_prefix = cond_str + f"discover_ROI-{roi_names[new_roi]}_"
    # print(cond_prefix)

    train_roi_ = train_data_rois[new_roi]
    test_roi_ = test_data_rois[new_roi]

    n_train = train_roi_.shape()[0]
    roi_dim = train_roi_.shape()[1]
    n_test = test_roi_.shape()[0]
    # print(n_train, n_test, roi_dim)
    # zero padding
    if n_latent <= 0:
        n_dim = roi_dim

        if model["n_feature"] > n_dim:
            n_pad_dim = model["n_feature"] - n_dim
            train_roi_.data = np.concatenate([train_roi_.data, np.zeros([n_train, n_pad_dim])], axis=1)
            test_roi_.data = np.concatenate([test_roi_.data, np.zeros([n_test, n_pad_dim])], axis=1)
            n_dim = model["n_feature"]
    else:
        n_dim = n_latent
        ## done: n_latent > roi_dim

    train_data = train_roi_
    test_data = test_roi_

    aligned_model = train_data.align(model["alignment"]["common_model"], method=model["method"])  # Q. n_feature?
    # print(aligned_model)
    # print(aligned_model.keys())
    # common_model_diff = np.sum(np.abs(model["alignment"]["common_model"] - aligned_model["common_model"]))
    # print(common_model_diff)  # 0.0
    # Q. Is the output common model the same as input common model?

    # print(model["alignment"]["common_model"].shape)
    train_transformed = aligned_model['transformed']
    # print(train_transformed.shape)
    train = [train_transformed, model["alignment"]["common_model"]]  # is this target correct?
    # print(train)
    irc_train, irc_train_mean = calc_irc(train, model["n_feature"])  # n_feature
    # print(list(irc_train.values()))
    # irc_train = list(aligned_model['isc'].values())
    # print(irc_train)

    # print(test_mean.shape)
    new_roi_p = aligned_model['transformation_matrix'].data.T
    test_transformed = np.dot(test_data.data, new_roi_p)
    # print(test_transformed.shape)
    # test = [test_transformed, model["alignment"]["common_model"]]  # is this target correct?
    test = [test_transformed, test_mean]
    # print(test)
    irc_test, irc_test_mean = calc_irc(test, model["n_feature"])
    # print(list(irc_test.values()))

    return float(irc_train_mean), float(irc_test_mean)


def draw_heatmap_discovery(data, labels, figtitle="test"):
    n_label = len(labels)
    label_width = max(1, n_label - 2)
    fig_scaler = 1  # tmp
    fig = plt.figure(figsize=(int(fig_scaler*n_label/3), int(fig_scaler*n_label/3)))  # heuristic figsize
    spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[n_label-label_width, label_width])
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])
    data_max_idx = np.argmax(data)
    data_sort_idx = np.argsort(data.reshape(-1))[::-1]
    data_sorted = data[data_sort_idx, :]
    labels_sorted = [labels[dsi] for dsi in data_sort_idx]
    heatmap0 = ax0.pcolor(data_sorted, cmap="viridis")
    # pp0 = fig.colorbar(heatmap0, ax=ax0, orientation="vertical", location='left')
    #  pp0 = fig.colorbar(heatmap0, ax=ax0, location='left')
    pp0 = fig.colorbar(heatmap0, ax=ax0, orientation='horizontal')
    ax0.set_yticks(np.arange(n_label) + 0.5, minor=False)
    ax0.set_xticks([])
    ax0.invert_yaxis()
    ax0.yaxis.tick_right()
    ax0.set_yticklabels(labels_sorted, minor=False)
    ax0.set_xticklabels([])
    ax0.set_title("(Sorted)")
    ax1.axis('off')

    # print(figtitle)
    # plt.show()
    plt.savefig(figtitle + "_discover_srm.png")
    plt.close()
