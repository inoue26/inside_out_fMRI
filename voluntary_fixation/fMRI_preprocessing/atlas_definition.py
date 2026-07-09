import os as _os
# atlas/ はこのファイルと同じディレクトリにあるので、CWD に依存せず __file__ から解決する
_ATLAS_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'atlas')

parc_path = _os.path.join(_ATLAS_DIR, 'HCP-MMP1_on_MNI152_ICBM2009c_nlin.nii.gz')  # 88616 voxels in total out of 238955
parc_path_lr = _os.path.join(_ATLAS_DIR, 'HCP-MMP1_on_MNI152_ICBM2009c_nlin_lr.nii.gz')
parc_lh_path = _os.path.join(_ATLAS_DIR, 'lh.HCP-MMP1.annot')
parc_rh_path = _os.path.join(_ATLAS_DIR, 'rh.HCP-MMP1.annot')

ROIs_22 = [
    {"name": "Primary Visual Cortex", "idxs": [1]},  # 2422
    {"name": "Early Visual Cortex", "idxs": [4, 5, 6]},  # 4579
    {"name": "Dorsal Stream Visual Cortex", "idxs": [13, 19, 3, 152, 16, 17]},  # 1413
    {"name": "Ventral Stream Visual Cortex", "idxs": [7, 163, 22, 18, 153, 160, 154]},  # 3214
    {"name": "MT+ Complex and Neighboring Visual Areas", "idxs": [158, 20, 21, 159, 156, 157, 23, 2, 138]},  # 2115
    {"name": "Somatosensory and Motor Cortex", "idxs": [8, 53, 9, 51, 52]},  # 4565
    {"name": "Paracentral Lobular and Mid Cingulate Cortex", "idxs": [40, 41, 55, 44, 43, 36, 39, 37]},  # 3906
    {"name": "Premotor Cortex", "idxs": [12, 54, 96, 10, 56, 78, 11]},  # 3286
    {"name": "Posterior Opercular Cortex", "idxs": [99, 113, 100, 101, 102, 105]},  # 2255
    {"name": "Early Auditory Cortex", "idxs": [24, 174, 173, 124, 104]},  # 984
    {"name": "Auditory Association Cortex", "idxs": [175, 129, 128, 130, 176, 123, 107]},  # 2918
    {"name": "Insular and Frontal Opercular Cortex", "idxs": [103, 178, 168, 167, 106, 115, 114, 109, 111, 112, 110, 108, 169]},  # 4117
    {"name": "Medial Temporal Cortex", "idxs": [120, 119, 118, 122, 126, 155, 127]},  # 3251
    {"name": "Lateral Temporal Cortex", "idxs": [137, 133, 177, 132, 136, 134, 172, 131, 135]},  # 8107
    {"name": "Temporo-Parieto-Occipital Junction", "idxs": [139, 140, 141, 28, 25]},  # 2138
    {"name": "Superior Parietal Cortex", "idxs": [48, 95, 49, 117, 50, 47, 42, 45, 46, 29]},  # 2965
    {"name": "Inferior Parietal Cortex", "idxs": [143, 151, 150, 149, 148, 116, 147, 146, 145, 144]},  # 6340
    {"name": "Posterior Cingulate Cortex", "idxs": [142, 121, 31, 15, 14, 33, 34, 35, 161, 162, 32, 38, 27, 30]},  # 5561
    {"name": "Anterior Cingulate and Medial Prefrontal Cortex", "idxs": [58, 57, 59, 180, 61, 60, 179, 62, 64, 165, 63, 69, 88, 65, 164]},  # 6585
    {"name": "Orbital and Polar Frontal Cortex", "idxs": [94, 66, 77, 91, 92, 89, 170, 90, 72, 93, 166]},  # 6477
    {"name": "Inferior Frontal Cortex", "idxs": [74, 75, 80, 79, 81, 82, 76, 171]},  # 2913
    {"name": "DorsoLateral Prefrontal Cortex", "idxs": [73, 67, 97, 98, 26, 70, 71, 87, 68, 83, 85, 84, 86]},  # 7722
]
ROIs_22_L = []
for roi in ROIs_22:
    roi_L = {"name": "L " + roi["name"], "idxs": roi["idxs"]}
    ROIs_22_L.append(roi_L)
ROIs_22_R = []
for roi in ROIs_22:
    roi_R = {"name": "R " + roi["name"], "idxs": [r + 180 for r in roi["idxs"]]}
    ROIs_22_R.append(roi_R)
ROIs_22_LR = ROIs_22_L + ROIs_22_R

ROIs_22_sub = [ROIs_22[0], ROIs_22[9]]

ROIs_22_dummy = [
    {"name": ROIs_22[0]["name"], "idxs": [1]},
    {"name": ROIs_22[1]["name"], "idxs": [2]},
    {"name": ROIs_22[2]["name"], "idxs": [3]},
    {"name": ROIs_22[3]["name"], "idxs": [4]},
    {"name": ROIs_22[4]["name"], "idxs": [5]},
    {"name": ROIs_22[5]["name"], "idxs": [6]},
    {"name": ROIs_22[6]["name"], "idxs": [7]},
    {"name": ROIs_22[7]["name"], "idxs": [8]},
    {"name": ROIs_22[8]["name"], "idxs": [9]},
    {"name": ROIs_22[9]["name"], "idxs": [10]},
    {"name": ROIs_22[10]["name"], "idxs": [11]},
    {"name": ROIs_22[11]["name"], "idxs": [12]},
    {"name": ROIs_22[12]["name"], "idxs": [13]},
    {"name": ROIs_22[13]["name"], "idxs": [14]},
    {"name": ROIs_22[14]["name"], "idxs": [15]},
    {"name": ROIs_22[15]["name"], "idxs": [16]},
    {"name": ROIs_22[16]["name"], "idxs": [17]},
    {"name": ROIs_22[17]["name"], "idxs": [18]},
    {"name": ROIs_22[18]["name"], "idxs": [19]},
    {"name": ROIs_22[19]["name"], "idxs": [20]},
    {"name": ROIs_22[20]["name"], "idxs": [21]},
    {"name": ROIs_22[21]["name"], "idxs": [22]},
]

ROIs_180 = [
    {"name": "V1, Primary Visual Cortex", "idxs": [1]},
    {"name": "MST, Medial Superior temporal Area", "idxs": [2]},
    {"name": "V6, Sixth Visual Area", "idxs": [3]},
    {"name": "V2, Second Visual Area", "idxs": [4]},
    {"name": "V3, Third Visual Area", "idxs": [5]},
    {"name": "V4, Fourth Visual Area", "idxs": [6]},
    {"name": "V8, Eight Visual Area", "idxs": [7]},
    {"name": "4, Primary Motor Cortex", "idxs": [8]},
    {"name": "3b, Primary Sensory Cortex", "idxs": [9]},
    {"name": "FEF, Frontal Eye Fields", "idxs": [10]},
    {"name": "PEF, Premotor Eye Field", "idxs": [11]},
    {"name": "55b, Area 55b", "idxs": [12]},
    {"name": "V3A, Area V3A", "idxs": [13]},
    {"name": "RSC, RetroSplenial Complex", "idxs": [14]},
    {"name": "POS2, Parieto-Occipital Sulcus Area 2", "idxs": [15]},
    {"name": "V7, Seventh Visual Area", "idxs": [16]},
    {"name": "IPS1, IntraParietal Sulcus Area 1", "idxs": [17]},
    {"name": "FFC, Fusiform Face Complex", "idxs": [18]},
    {"name": "V3B, Area V3B", "idxs": [19]},
    {"name": "LO1, Area Lateral Occipital 1", "idxs": [20]},
    {"name": "LO2, Area Lateral Occipital 2", "idxs": [21]},
    {"name": "PIT, Posterior InferoTemporal", "idxs": [22]},
    {"name": "MT, Middle Temporal Area", "idxs": [23]},
    {"name": "A1, Primary Auditory Cortex", "idxs": [24]},
    {"name": "PSL, PeriSylvian Language Area", "idxs": [25]},
    {"name": "SFL, Superior Frontal Language Area", "idxs": [26]},
    {"name": "PCV, PreCuneus Visual Area", "idxs": [27]},
    {"name": "STV, Superior Temporal Visual Area", "idxs": [28]},
    {"name": "7Pm, Medial Area 7P", "idxs": [29]},
    {"name": "7m, Area 7m", "idxs": [30]},
    {"name": "POS1, Parieto-Occipital Sulcus Area 1", "idxs": [31]},
    {"name": "23d, Area 23d", "idxs": [32]},
    {"name": "v23ab, Area ventral 23 a+b", "idxs": [33]},
    {"name": "d23ab, Area dorsal 23 a+b", "idxs": [34]},
    {"name": "31pv, Area 31p ventral", "idxs": [35]},
    {"name": "5m, Area 5m", "idxs": [36]},
    {"name": "5mv, Area 5m ventral", "idxs": [37]},
    {"name": "23c, Area 23c", "idxs": [38]},
    {"name": "5L, Area 5L", "idxs": [39]},
    {"name": "24dd, Dorsal Area 24d", "idxs": [40]},
    {"name": "24dv, Ventral Area 24d", "idxs": [41]},
    {"name": "7AL, Lateral Area 7A", "idxs": [42]},
    {"name": "SCEF, Supplementary and Cingulate Eye Field", "idxs": [43]},
    {"name": "6ma, Area 6m anterior", "idxs": [44]},
    {"name": "7Am, Medial Area 7A", "idxs": [45]},
    {"name": "7Pl, Lateral Area 7P", "idxs": [46]},
    {"name": "7PC, Area 7PC", "idxs": [47]},
    {"name": "LIPv, Area Lateral IntraParietal ventral", "idxs": [48]},
    {"name": "VIP, Ventral IntraParietal Complex", "idxs": [49]},
    {"name": "MIP, Medial IntraParietal Area", "idxs": [50]},
    {"name": "1, Area 1", "idxs": [51]},
    {"name": "2, Area 2", "idxs": [52]},
    {"name": "3a, Area 3a", "idxs": [53]},
    {"name": "6d, Dorsal area 6", "idxs": [54]},
    {"name": "6mp, Area 6mp", "idxs": [55]},
    {"name": "6v, Ventral Area 6", "idxs": [56]},
    {"name": "p24pr, Area Posterior 24 prime", "idxs": [57]},
    {"name": "33pr, Area 33 prime", "idxs": [58]},
    {"name": "a24pr, Anterior 24 prime", "idxs": [59]},
    {"name": "p32pr, Area p32 prime", "idxs": [60]},
    {"name": "a24, Area a24", "idxs": [61]},
    {"name": "d32, Area dorsal 32", "idxs": [62]},
    {"name": "8BM, Area 8BM", "idxs": [63]},
    {"name": "p32, Area p32", "idxs": [64]},
    {"name": "10r, Area 10r", "idxs": [65]},
    {"name": "47m, Area 47m", "idxs": [66]},
    {"name": "8Av, Area 8Av", "idxs": [67]},
    {"name": "8Ad, Area 8Ad", "idxs": [68]},
    {"name": "9m, Area 9 Middle", "idxs": [69]},
    {"name": "8BL, Area 8B Lateral", "idxs": [70]},
    {"name": "9p, Area 9 Posterior", "idxs": [71]},
    {"name": "10d, Area 10d", "idxs": [72]},
    {"name": "8C, Area 8C", "idxs": [73]},
    {"name": "44, Area 44", "idxs": [74]},
    {"name": "45, Area 45", "idxs": [75]},
    {"name": "47l, Area 47l (47 lateral)", "idxs": [76]},
    {"name": "a47r, Area anterior 47r", "idxs": [77]},
    {"name": "6r, Rostral Area 6", "idxs": [78]},
    {"name": "IFJa, Area IFJa", "idxs": [79]},
    {"name": "IFJp, Area IFJp", "idxs": [80]},
    {"name": "IFSp, Area IFSp", "idxs": [81]},
    {"name": "IFSa, Area IFSa", "idxs": [82]},
    {"name": "p9-46v, Area posterior 9-46v", "idxs": [83]},
    {"name": "46, Area 46", "idxs": [84]},
    {"name": "a9-46v, Area anterior 9-46v", "idxs": [85]},
    {"name": "9-46d, Area 9-46d", "idxs": [86]},
    {"name": "9a, Area 9 anterior", "idxs": [87]},
    {"name": "10v, Area 10v", "idxs": [88]},
    {"name": "a10p, Area anterior 10p", "idxs": [89]},
    {"name": "10pp, Polar 10p", "idxs": [90]},
    {"name": "11l, Area 11l", "idxs": [91]},
    {"name": "13l, Area 13l", "idxs": [92]},
    {"name": "OFC, Orbital Frontal Complex", "idxs": [93]},
    {"name": "47s, Area 47s", "idxs": [94]},
    {"name": "LIPd, Area Lateral IntraParietal dorsal", "idxs": [95]},
    {"name": "6a, Area 6 anterior", "idxs": [96]},
    {"name": "i6-8, Inferior 6-8 Transitional Area", "idxs": [97]},
    {"name": "s6-8, Superior 6-8 Transitional Area", "idxs": [98]},
    {"name": "43, Area 43", "idxs": [99]},
    {"name": "OP4, Area OP4/PV", "idxs": [100]},
    {"name": "OP1, Area OP1/SII", "idxs": [101]},
    {"name": "OP2-3 Area OP2-3/VS", "idxs": [102]},
    {"name": "52, Area 52", "idxs": [103]},
    {"name": "RI, RetroInsular Cortex", "idxs": [104]},
    {"name": "PFcm, Area PFcm", "idxs": [105]},
    {"name": "PoI2, Posterior Insular Area 2", "idxs": [106]},
    {"name": "TA2, Area TA2", "idxs": [107]},
    {"name": "FOP4, Frontal OPercular Area 4", "idxs": [108]},
    {"name": "MI, Middle Insular Area", "idxs": [109]},
    {"name": "Pir, Pirform Cortex", "idxs": [110]},
    {"name": "AVI, Anterior Ventral Insular Area", "idxs": [111]},
    {"name": "AAIC, Anterior Agranular Insula Complex", "idxs": [112]},
    {"name": "FOP1, Frontal OPercular Area 1", "idxs": [113]},
    {"name": "FOP3, Frontal OPercular Area 3", "idxs": [114]},
    {"name": "FOP2, Frontal OPercular Area 2", "idxs": [115]},
    {"name": "PFt, Area PFt", "idxs": [116]},
    {"name": "AIP, Anterior IntraParietal Area", "idxs": [117]},
    {"name": "EC, Entorhinal Cortex", "idxs": [118]},
    {"name": "PreS, PreSubiculum", "idxs": [119]},
    {"name": "H, Hippocampus", "idxs": [120]},
    {"name": "ProS, ProStriate Area", "idxs": [121]},
    {"name": "PeEc, Perirhinal Ectorhinal Cortex", "idxs": [122]},
    {"name": "STGs, Area STGa", "idxs": [123]},
    {"name": "PBelt, ParaBelt Complex", "idxs": [124]},
    {"name": "A5, Auditory 5 Complex", "idxs": [125]},
    {"name": "PHA1, ParaHippocampal Area 1", "idxs": [126]},
    {"name": "PHA3, ParaHippocampal Area 3", "idxs": [127]},
    {"name": "STSda, Area STSd anterior", "idxs": [128]},
    {"name": "STSdp, Area STSd posterior", "idxs": [129]},
    {"name": "STSvp, Area STSv posterior", "idxs": [130]},
    {"name": "TGd, Area TG dorsal", "idxs": [131]},
    {"name": "TE1a, Area TE1 anterior", "idxs": [132]},
    {"name": "TE1p, Area TE1 posterior", "idxs": [133]},
    {"name": "TE2a, Area TE2 anterior", "idxs": [134]},
    {"name": "TF, Area TF", "idxs": [135]},
    {"name": "TE2p, Area TE2 posterior", "idxs": [136]},
    {"name": "PHT, Area PHT", "idxs": [137]},
    {"name": "PH, Area PH", "idxs": [138]},
    {"name": "TPOJ1, Area TemporoParietoOccipital Junction 1", "idxs": [139]},
    {"name": "TPOJ2, Area TemporoParietoOccipital Junction 2", "idxs": [140]},
    {"name": "TPOJ3, Area TemporoParietoOccipital Junction 3", "idxs": [141]},
    {"name": "DVT, Dorsal Transitional Visual Area", "idxs": [142]},
    {"name": "PGp, Area PGp", "idxs": [143]},
    {"name": "IP2, Area IntraParietal 2", "idxs": [144]},
    {"name": "IP1, Area IntraParietal 1", "idxs": [145]},
    {"name": "IP0, Area IntraParietal 0", "idxs": [146]},
    {"name": "PFop, Area PF opercular", "idxs": [147]},
    {"name": "PF, Area PF Complex", "idxs": [148]},
    {"name": "PFm, Area PFm Complex", "idxs": [149]},
    {"name": "PGi, Area PGi", "idxs": [150]},
    {"name": "PGs, Area PGs", "idxs": [151]},
    {"name": "V6A, Area V6A", "idxs": [152]},
    {"name": "VMV1, VentroMedial Visual Area 1", "idxs": [153]},
    {"name": "VMV3, VentroMedial Visual Area 3", "idxs": [154]},
    {"name": "PHA2, ParaHippocampal Area 2", "idxs": [155]},
    {"name": "V4t, Area V4t", "idxs": [156]},
    {"name": "FST, Area FST", "idxs": [157]},
    {"name": "V3CD, Area V3CD", "idxs": [158]},
    {"name": "LO3, Area Lateral Occipital 3", "idxs": [159]},
    {"name": "VMV2, VentroMedial Visual Area 2", "idxs": [160]},
    {"name": "31pd, Area 31pd", "idxs": [161]},
    {"name": "31a, Area 31a", "idxs": [162]},
    {"name": "VVC, Ventral Visual Complex", "idxs": [163]},
    {"name": "25, Area 25", "idxs": [164]},
    {"name": "s32, Area s32", "idxs": [165]},
    {"name": "pOFC, posterior OFC Complex", "idxs": [166]},
    {"name": "PoI1, Area Posterior Insular 1", "idxs": [167]},
    {"name": "Ig, Insular Granular Complex", "idxs": [168]},
    {"name": "FOP5, Area Frontal Opercular 5", "idxs": [169]},
    {"name": "p10p, Area posterior 10p", "idxs": [170]},
    {"name": "p47r, Area posterior 47r", "idxs": [171]},
    {"name": "TGv, Area TG Ventral", "idxs": [172]},
    {"name": "MBelt, Medial Belt Complex", "idxs": [173]},
    {"name": "LBelt, Lateral Belt Complex", "idxs": [174]},
    {"name": "A4, Auditory 4 Complex", "idxs": [175]},
    {"name": "STSva, Area STSv anterior", "idxs": [176]},
    {"name": "TE1m, Area TE1 Middle", "idxs": [177]},
    {"name": "PI, Para-Insular Area", "idxs": [178]},
    {"name": "a32pr, Area anterior 32 prime", "idxs": [179]},
    {"name": "p24, Area posterior 24", "idxs": [180]},
]
ROIs_180_L = []
for roi in ROIs_180:
    roi_L = {"name": "L " + roi["name"], "idxs": roi["idxs"]}
    ROIs_180_L.append(roi_L)
ROIs_180_R = []
for roi in ROIs_180:
    roi_R = {"name": "R " + roi["name"], "idxs": [r + 180 for r in roi["idxs"]]}
    ROIs_180_R.append(roi_R)
ROIs_180_LR = ROIs_180_L + ROIs_180_R

ROIs_180_sub = [ROIs_180[0], ROIs_180[23]]
