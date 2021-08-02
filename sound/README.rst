Generate Features
=================
Use prepare_dataset.py to generate the MFCCs of the dataset. For example for
the Speech Commands dataset (1-sec clips at 16kHZ), 40 mel-bands, a step
of 10ms, and a window length of 25ms the script should be called::

   prepare_dataset mfccs <dataset_path> 16000 40 400 160

A json file will be generated from this script containing the MFCCs of the
whole dataset in <dataset_path>.

Run Experiments
===============
Give the name of the json file that was generated to the DATA_PATH constant in
run_global.py and run the script.
