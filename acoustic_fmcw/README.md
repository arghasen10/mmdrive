# Acoustic FMCW 

## Directory Structure
```
mmdrive
└── acoustic_fmcw
    └── Android
    └── post_process
    └── RF_Classifier
        └── amp_phase.pkl  
        └── Aqu_dop.pkl  
        └── rf_aqu_amp-dvn.py  
        └── rf_aqu_amp.py  
        └── rf_aqu_rd.py
    └── README.md
```

Source code for the Android Application is provided in **Android directory. The role of this application is to transmit and receive FMCW chirps at ultrasound frequency range (16kHz-19kHz). Chirp data is stored locally in the Andoid phone.

Once the data collection phase ends we post process this data using the scripts provided in **post_process** directory. It performs 1D-FFT on the raw chirp data to retrieve amplitude and phase across different range bins. In the next step we perform 2D-FFT on the range bins to retrieve range-doppler heatmap. Finally all these processed data is provided to a RandomForest Classifier to classify different dangerous driving behaviors. Source code for the Random Forest Classifier is provided in the **RF_Classifier** directory.     