#!/bin/sh

FACIAL_FOLDER=~/Development/FACIAL
DATA_FOLDER=~/Development/facial_data
VIDEO_PREPROCESS_FOLDER=$DATA_FOLDER/video_preprocess
AUDIO_PREPROCESS_FOLDER=$DATA_FOLDER/audio_preprocessed


init() {
  echo "init..."
  pip install --upgrade --no-cache-dir gdown
  deepspeech_model=$FACIAL_FOLDER/audio2face/ds_graph/output_graph.pb
  if [ ! -f $deepspeech_model ]
  then
    # Deepspeech 0.1.0 model
    gdown -O $deepspeech_model  https://drive.google.com/uc?id=1xsqUpHG6H8UJz0WB6Tp9xpQ1xYvmxuMK
  fi

  # Build face3d
  cd $FACIAL_FOLDER/face_render/face3d/mesh/cython
  python setup.py build_ext -i

  bfm_folder=$FACIAL_FOLDER/face_render/BFM
  if [ ! -d $bfm_folder ]
  then
    # You have to download the BFM09 model and put it into /content/FACIAL/face_render/BFM subfolder. Following the instruction with https://github.com/microsoft/Deep3DFaceReconstruction
    mkdir $bfm_folder
    gdown -O $bfm_folder/BFM_model_front.mat https://drive.google.com/uc?id=1Lp2UP4N5gBf26d_4IomIhyrlbZCV4Tvp
    gdown -O $bfm_folder/std_exp.txt https://drive.google.com/uc?id=1V4ZtHcTMURQDJucCqpmSJtCKPBgcu5hN
  fi
}

face_render() {
  echo "face render..."
  cd $FACIAL_FOLDER/face_render
  python handle_netface.py --param_folder $VIDEO_PREPROCESS_FOLDER/train1_deep3Dface

  python fit_headpose.py --csv_path $VIDEO_PREPROCESS_FOLDER/train1_openface/train1_512_audio.csv \
    --deepface_path $VIDEO_PREPROCESS_FOLDER/train1_deep3Dface/train1.npz \
    --save_path $VIDEO_PREPROCESS_FOLDER/train1_posenew.npz

  python render_netface_fitpose.py --real_params_path $VIDEO_PREPROCESS_FOLDER/train1_posenew.npz \
    --outpath $VIDEO_PREPROCESS_FOLDER/train_A
}

audio_preprocess(){
  echo "audio preprocess..."
  if [ ! -f "$VIDEO_PREPROCESS_FOLDER/train1.wav" ]
  then
    ffmpeg -i $VIDEO_PREPROCESS_FOLDER/train1.mp4 -acodec pcm_s16le -f wav -ac 1 -ar 16000  $VIDEO_PREPROCESS_FOLDER/train1.wav
  fi
  cd $FACIAL_FOLDER/audio2face
  python audio_preprocessing.py --input_file $VIDEO_PREPROCESS_FOLDER/train1.wav --output_path $AUDIO_PREPROCESS_FOLDER
}

train_audio_to_face(){
  echo "training audio to face"
  if [ ! -f "$DATA_FOLDER/audio2face/checkpoint/obama" ]
  then
    mkdir -p $DATA_FOLDER/audio2face/checkpoint/obama
    gdown -O $DATA_FOLDER/audio2face/checkpoint/obama/Gen-20-0.0006273046686902202.mdl https://drive.google.com/uc?id=1NN5aH_R_OZFrCIQji1j-bpb5AETocBnf
  fi

  if [ ! -f "$DATA_FOLDER/audio2face/data" ]
  then
    mkdir -p $DATA_FOLDER/audio2face/data
    gdown -O $DATA_FOLDER/audio2face/data/train3.npz https://drive.google.com/uc?id=14mQo7U7VcwWxQ4y6EOC4vchmJBjbTYwB
  fi

  cd $FACIAL_FOLDER/audio2face
  python fintuning2-trainheadpose.py --audiopath $AUDIO_PREPROCESS_FOLDER/train1.pkl \
        --npzpath $VIDEO_PREPROCESS_FOLDER/train1_posenew.npz \
        --cvspath $VIDEO_PREPROCESS_FOLDER/train1_openface/train1_512_audio.csv \
        --pretainpath_gen $DATA_FOLDER/audio2face/checkpoint/obama/Gen-20-0.0006273046686902202.mdl \
        --savepath $DATA_FOLDER/audio2face/checkpoint/train1
}

train_face_to_vid(){

  if [ ! -d "$DATA_FOLDER/face2vid/train_A" ]
  then
    cp -r $VIDEO_PREPROCESS_FOLDER/train_A $DATA_FOLDER/face2vid/train_A
  fi

  if [ ! -d "$DATA_FOLDER/face2vid/train_B" ]
  then
    cp -r $VIDEO_PREPROCESS_FOLDER/train1_image $DATA_FOLDER/face2vid/train_B
  fi


  #  pip uninstall -y scipy
  #  pip install scipy==1.2.0
  cd $FACIAL_FOLDER/face2vid
  python train.py --checkpoints_dir $DATA_FOLDER/checkpoints \
  --blink_path $VIDEO_PREPROCESS_FOLDER/train1_openface/train1_512_audio.csv \
  --name train3 --model pose2vid --dataroot $DATA_FOLDER/face2vid/ --netG local --ngf 32 --num_D 3 \
  --tf_log --niter_fix_global 0 --label_nc 0 --no_instance --save_epoch_freq 2 --lr=0.0001 \
  --resize_or_crop resize --no_flip --verbose --n_local_enhancers 1
}

#init
#face_render
#audio_preprocess
#train_audio_to_face
train_face_to_vid