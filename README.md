# Project Borealis 

### Abstract
Today, the latest cutting-edge for user-oriented commercial millimeter-wave (mmwave) RADAR solutions utilize a Pulse Modulated Continuous Wave (PMCW) transmission signal to obtain useful insight into the area surrounding a spacial region. These RADAR systems can measure near-instantaneous and highly accurate range information (range resolutions up to a nanometer-level), doppler information (resolutions up to a few cm/second), and angular information in both azimuth and elevation. Coupled with RADAR's innate robustness against weather, lighting, and interference from other RADARs, this sensing modality presents itself as a unique means of complementing existing LiDAR and Camera based autonomous driving capabilities.

Needless to say, the safety of both passengers and passerbys is mission critical. Therefore, it's vital that autonomous systems use their wide array of sensors to detect, track, and classify the types of objects in a scene. If this can be done using only RADAR sensors, correct classification of moving objects can be done irrespective of uncontrollable circumstances like weather and lighting. In this thesis, classification of detections as pedestrian, bicyle, and car is achieved using only the range-doppler information from a single forward facing autonomotive PMCW RADAR system.

**Keywords**: rf, radar, PMCW, DCM, classification

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
Increasingly, autonomous driving is moving closer to reality. Over the last few years, automotive and technology companies alike have made major progress in putting these systems on geo-fenced roads that everyday Americans drive. However, in order to bring about a true level-5 autonomous transportation revolution, improvements need to be made to these autonomous vehicles such that they can be demonstratably safe in all possible driving conditions. Currently, work is being done all across the perception stack; from high-level perception algorithms to low-level sensing hardware. Specifically relevant to this work are improvements to the RADAR sensing platform.

Historically, RADAR systems have provided robust detection capabilities in a wide range of weather conditions. This, coupled with its long range and fine resolution, made RADAR systems popular in military hardware. First developed in the 1930's, RADAR is widely considered to be a mature field of applied science \cite{https://www.nature.com/articles/156319a0#citeas}. However, as more advanced RADAR hardware finds itself new applications beyond aerial and naval detection/tracking, more sophisticated software and algorithms can be introduced to capitalize on the new capabilities. To meet these new application domains, consumer RADAR systems must often customize to their desired target use case. This has certainly been true for RADAR systems like Google's Soli RADAR chip, which was specialized for human computer interaction and has debuted on the Google Pixel 4 smartphone platform \cite{soli_radar}. The same can be said for RADAR systems destined for use in autonomous vehicles.

Recently, a start-up company called Uhnder has released a novel RADAR sensing platform to the wider autonomous vehicle market. The RADAR system, known as Judo, uses pulse modulated continuous wave (PMCW) transmission signals to build up the radar data cube (RDC) \cite{DCM_Uhnder}. Traditionally, frequency modulated continuous wave (FMCW) transmission signals are used to the same ends. By moving to a PMCW based system, the Judo module makes improvements to range contrast resolution (good autocorrelation properties between transmitted codes), angular resolution (large virtual receiver count), and native rejection of other background EM signals occupying the same frequencies. These improvements are all extremely useful for the end application of providing perception insights for general autonomous driving.

Between Uhnder's Judo module and Google's Soli chip, it's clear that RADAR can be used to perform detection/tracking in automotive use cases as well as classify complex hand gestures. Would it be possible to bring gesture classification to the automotive RADAR domain? This is particularly interesting because whole body motions like walking, running, and bike riding can be modeled as whole-body human gestures. Making this capability a reality would allow an additional senor on the autonomous vehicle sensor suite to distill a useful perception insight in all weather and lighting conditions.

In this work, I present an analysis of common classification techniques performance on distinguishing between running pedestrians, moving bicycles, and moving cars using only information gathered from an automotive RADAR system. Specifically, classification is done on the range-doppler data gathered using a Judo RADAR system.


## Literature Review
### Other Sensor based Classification
The problem of classifying moving roadway objects as pedestrians, bicycles and cars is an integral part of any autonomous vehicle. As the predominant imaging sensors on such platforms are generally a mixture of cameras and LiDAR systems, it's no surprise that work has been done using these sensing systems. Using sensor fusion between a camera and a LiDAR, separate classifiers and a fusion method are trained together to achieve about 88% accuracy \cite{https://www.semanticscholar.org/paper/LIDAR-and-vision-based-pedestrian-detection-system-Premebida-Ludwig/7dd429f800640815bf56c7834493321a3970e3ea}. As LiDAR sensors have improved, work has also been done on using only LiDARs to detect multiple bicycles in a scene \cite{https://repository.library.northeastern.edu/files/neu:cj82qt54g/fulltext.pdf}. While the qualitative results of the classification scheme developed in the previous work show promise, there is clearly more work that needs to be done to address some of the situations where the liDAR has issues (i.e. bike near car, multiple bikes close together in scene, etc).

### RADAR based Classification
The concept of classifying objects based on their RADAR signature is not new. In previous works, the microdoppler signatures (velocity vs. time) of moving objects has been examined both in simulation \cite{uDoppler_matlab} and in practice academic research \cite{https://ro.uow.edu.au/cgi/viewcontent.cgi?article=2605&context=eispapers}. Discrimination accuracy between these two categories of signals can, at times, be as high as 95% \cite{CS 598 Final Project}. This poses some practical challenges, however. Since one of the axes is time itself, it's difficult to perform this analysis in real time. Additionally, a lot of the work previously done used 24GHz FMCW RADAR sensors. By moving to a 70GHz PMCW RADAR system, the discretionary power of the RADAR should increase with the improvements to high contrast resolution.

Alternatively, the problem can be approached using the range doppler plots over time. Using this view of the RDC, it's easier to build models like hidden markov models (HMM) and recurrent neural networks (RNN) which leverage temporal information to classify gestures. Using range doppler plots, the Soli team at Google ATAP has proposed a CNN+RNN based classification model to discern between 11 different types of micro hand gestures \cite{soli_classification}. Their model performs well at the given set of hand gestures with accuracies between 85% to 94% on their end-to-end CNN_RNN classification model architecture. This approach is possible because of the design of the Soli chip, which was designed specifically to greatly increase the RADAR sensor's temporal resolution. This choice was made knowing that, at that close range, the range resolution of a radar cannot compare with that of a 3D depth camera. Additionally, the spacial azimuth and elevation resolution of a RADAR can only currently be significantly improved by adding more antennas. This will consequently enlarge the physical size of the device; something Soli did not want to do on their single chip solution. For the autonomous vehicle application, RADAR systems are designed with designed mainly for target detection and tracking. Consequently, classification on a platform like this must use hardware built for more than just classification purposes. This jack of all trades system does not have a high temporal, range, doppler, or angular resolution. Instead, using a RADAR designed for the commercial automotive autonomy use case demonstrates the general feasibility of target classification using only RDC data.


## RADAR Fundamentals
 

### General Background
### Frequency-Modulated-Continuous-Wave (FMCW)
### Pulse-Modulated-Continuous-Wave (PMCW)

## Doppler Classification

## Experimental Design
### Judo Module Radar Description
### RDC1 to RDC2
### Classification

## Experimental Results
### Results
### Analysis
No one-size-fits-all
### References

https://www.nature.com/articles/156319a0#citeas