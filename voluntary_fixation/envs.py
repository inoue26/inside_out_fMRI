# general
SAVE_ROOT = '../../results/voluntary_fixation'
TMP_SAVE_ROOT = '../../results/voluntary_fixation_public'
DEVICE = 'cuda'
SUBJECT_IDS = ['01', '02', '03', '04', '06', '10', '14', '15', '16', '17', '18', '19']
RUN_IDS = [1, 2, 3, 4, 5, 6, 7, 8]


# movie info
MOVIE_FPS=25
MOVIE_WIDTH=1280
MOVIE_HEIGHT=720
SEGMENT_MOVIE_DIR = '../../original_movies'
TARGET_NUM_FRAMES = [22550, 22050, 21900, 24400, 23100, 21950, 27100, 16900] # for compatibility with VOLUMS, last one shoud be 16900, but authors' MLT output has 16876 frames.
BRIGHTNESS_DIR = '../../datalad/DataLad-101/studyforrest-data-confoundsannotation/annotation/visual'

# FMRI info
RUN_VOLUMES=[451, 441, 438, 488, 462, 439, 542, 338]
TR=2
LABEL_ROOT='./voluntary_fixation/dataset/labels'
BOLD44_ROOT='../../data/forrestgump/studyforrest/roi22LR_np'
NUM_ROIS=44

# Eye tracking info
EYE_TRACKING_FPS=1000
EYEMOVE_ROOT = '../../data/forrestgump/studyforrest/annot/studyforrest-data-eyemovementlabels'
COLOR_AREA_Y = (87, 633)

RUN_MODES = {
    0:[2,3,4,6,7,8], # train_runs
    1:[1,3,4,5,7,8],
    2:[1,2,4,5,6,8],
    3:[1,2,3,5,6,7],
    4:[3,4,5,6,7,8],
    5:[1,2,5,6,7,8],
    6:[1,2,3,4,7,8],
    7:[1,2,3,4,5,6],
    8:[2,3,4,5,6,7,8],
    9:[1,3,4,5,6,7,8],
    10:[1,2,4,5,6,7,8],
    11:[1,2,3,5,6,7,8],
    12:[1,2,3,4,6,7,8],
    13:[1,2,3,4,5,7,8],
    14:[1,2,3,4,5,6,8],
    15:[1,2,3,4,5,6,7],
}