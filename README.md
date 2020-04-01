# Project Borealis 

### Abstract
Today, the latest cutting-edge for user-oriented commercial millimeter-wave (mmwave) radar solutions utilize a Pulse Modulated Continuous Wave (PMCW) transmission signal to obtain highly accurate range information (range resolutions up to a nanometer-level). Conveniently, the same information used to estimate range can, when viewed over multiple modulation cycles, also allow the radar sensor to estimate target velocity (doppler). Viewing frame by frame range-doppler plots allows these systems to perform classification of complex human gestures. 

In the past, gesture classification has been used for small hand gesture classification. In this thesis, 

One possible application is to use the RADAR system's gesture classification capabilities to classify moving objects as pedestrians, bicycles, or cars. 

Over the last few years, a new class of mmwave radar systems utilizing pulse modulated continuous wave (PMCW) transmission signals have arrived on the consumer market. Since 


Since a PMCW chip can exhibit a period much shorter than a standard FMCW chirp, these new radar systems enable much finer range and doppler resolutions (range resolutions at the millimeter-level). Consequently, classification of more sophisticated and subtle gestures should be able to be achieved. This senior research thesis explores the human gesture recognition capabilities of these new PMCW radar systems compared to that of the FMCW radars that preceded it.

### Directory Structure
    .
    ├── data                    # Data directory
        ├── processed           # Processed data (RDC1 -> RDC2)
        ├── raw                 # Raw data (RDC1)
    ├── dataset                 # Scripts for that convert "raw" RDC1 data to "processed" RDC2 data
    ├── documents               # Various documents needed for and including the thesis component
    ├── models                  # Models for classification
    ├── .gitignore
    ├── README.md
    ├── requirements.txt        # Required dependencies to run this package.

### Getting Started
Welcome! Here are some steps to getting started.
1. Download [GitHub Desktop](https://desktop.github.com/)
2. Click "Add"
3. Click "Clone Repository..."
4. Click URL
5. Put (https://github.com/edwin-pan/Borealis.git) into the first box
6. Put whichever directory you'd like in the second box. 
7. Click "Clone"


## Introduction
Human communication manifests itself in many forms. In order to interact with each other, we most often employ our voices to speak, the written word to record messages, and body movement to augment the ways we express ourselves. As we adopt the move towards "smart" devices for our everyday machines, it's imperative that we build out robust systems that allow us to use all of our existing modes of communication to interact with our machines. \\
Naturally, the microphone's ability to measure incoming sound waves make this sensor well suited to the task of digitizing the human voice. Similarly, a camera's ability to capture high definition colored 2D spatial projections of our 3D environment allows it to be used in image classification (and thus text reading) with high accuracy. For body gestures, humans can gleam this understanding from a temporal sequence of colored 2D images of the 3D environment. However, while the human sensory input might be images, messages are communicated simply because we understand 'what' is moving, 'how' it is moving, and ignore the rest of the extraneous information. As a computer vision task, deciding 'what' is important in a scene and 'how' important things move is itself an ongoing field of study. Just estimating a starting and ending location in a 3D environment from the 2D image is a complex task. Thus, in order to classify human gestures, one option is to look for a more direct means of gathering the useful information without the extraneous data.

Since we desire to know accurate range information and near-instantaneous velocity information, the natural sensor to chose would be a RAdio Detection And Ranging (RADAR or radar) system. It's high range-resolution and high doppler-resolution as well as simple processing allow it to serve the goal of human gesture recognition well; despite it not being a core human sensory capability.

## Literature Review
Indeed, much work has been done in the past to explore the use of radar systems for this application. In 2016, Google unveiled a radar chip \cite{soli_radar}, designated Project Soli, dedicated solely to aim of classifying hand gestures. Later that year, a CNN+RNN neural network architecture was proposed to evaluate the feasibility of classifying between 11 different hand gestures on the Soli platform \cite{soli_classification}. In 2018, Texas Instruments demonstrated the capacity for their general purpose IWR1443 MIMO radar to recognize a simple twirl gesture at the annual Comsumer Electronics Show (CES). In late 2019, Google announced that the Soli radar chip would ship on the technology company's flagship Google Pixel 4 smartphone. The radar systems used in these developments were all FMCW radar systems. Around the same time, a new method for signal transmission was proposed using DCM techniques to obtain the same range differentiation \cite{DCM_Uhnder}. One major advantage of using modulated codes with beneficial autocorrelation properties is the significantly smaller transmission period $T_c$, integer transmission length $L_c$, and overall transmission time $T_cL_c$ needed to gather ranging information. Consequently, the range resolution improves greatly. Additionally, 


## RADAR Fundatmentals
### General Background
### Frequency-Modulated-Continuous-Wave (FMCW)
### Pulse-Modulated-Continuous-Wave (PMCW)

## Experimental Design
### Judo Module Radar Description
### RDC1 to RDC2
### Classification

## Experiemntal Results
### Results
### Analysis
### References
