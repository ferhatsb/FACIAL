#!/bin/sh

VIDEO_PREPROCESS_FOLDER=~/Development/FACIAL/video_preprocess

face_render() {
    echo "face render";
    cd ./face_render;
    python handle_netface.py --param_folder $VIDEO_PREPROCESS_FOLDER/train1_deep3Dface

    python fit_headpose.py --csv_path $VIDEO_PREPROCESS_FOLDER/train1_openface/train1_512_audio.csv \
      --deepface_path $VIDEO_PREPROCESS_FOLDER/train1_deep3Dface/train1.npz \
      --save_path $VIDEO_PREPROCESS_FOLDER/train1_posenew.npz

    python render_netface_fitpose.py --real_params_path $VIDEO_PREPROCESS_FOLDER/train1_posenew.npz \
      --outpath $VIDEO_PREPROCESS_FOLDER/train_A
    cd ..
}

audio_preprocess(){
  echo "audio preprocess";
  cd ./audio2face
  python audio_preprocessing.py --input_file $VIDEO_PREPROCESS_FOLDER/train.wav
  cd ..
}

face_render
audio_preprocess


