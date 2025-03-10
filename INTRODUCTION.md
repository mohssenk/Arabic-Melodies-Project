# Introduction to the Arabic Melodic Scales Classification Project


Welcome to my favorite and most sentimental project! In the following post, I will be explaining the background behind the niche genre of Arabic melodic scales, and my pioneering contribution to this ancient and prized part of Middle Eastern culture.


## What are the Arabic Melodic Scales (aka Maqams)?


At the basic level, a musical scale is a series of notes arranged in a structural order. These scales form structural outlines for melodies. From each scale, many melodies can be made. Each unique scale has a special rhythm or emotion that can be sensed in all the melodies that belong to it. 
Every culture has its musical traditions, and the Arab world is no different. In Arabic music, there are 8 famous scales used in music, singing, reading poems melodiously, and more. These 8 scales are also called the maqams. Each of these 8 is famously associated with an emotion.
For example, the scale of Hejaz induces a feeling of yearning and is often used in romance. It’s also incredibly iconic, being heavily associated with oriental music. In fact, the average person can often recognize its association with Middle Eastern culture.


In fact, here are all the 8 scales, their emotions, and associated examples. When listening, note which emotion it immediately begins to stir. While it is not needed to listen to any of them to them to understand this project, they are listed here for reference: 


  - [Hejaz](https://www.youtube.com/watch?v=v-OEgLDoKB4) - Yearing & awe
  - [Rast](https://youtu.be/hv3stJbTPCE?si=3eQtpRQU0ZoLlQMT) - Confidence & strength
  - [Bayat](https://youtu.be/MHk6Z_eFqXM?si=f1EomGXKNMv_ehqC) - Spiritual depth
  - [Nahawand](https://youtu.be/9czk_aQXNR0?si=G7kIfB2mgH77fYEs) - Tenderness (corresponds with the western minor scale)
  - [Ajam](https://youtu.be/ZpLr6BGR_fE?si=8eE2Yc-fuxIWCU4i) - Harmony & joy (corresponds with the western major scale)
  - [Saba](https://youtu.be/x7fGYEIhrhM?si=qAAdztLeGsqfKAEl) - Sorrow
  - [Kurd](https://youtu.be/NRuSxAjGOgo?si=oxG8nxWuR32otr7d) - Simplicity
  - [Seekah](https://youtu.be/hmby0lm1DfA?si=V9bM_pM4l32RZ8l8) - Intimacy & devotion


These 8 scales aren’t just used for musical instruments, but for singing and reading poems melodiously. Melodiously orating long, well-written poems is a very large part of Arabic culture, and the 8 scales come heavily in handy. One can choose a scale to use while orating their poem or script, depending on the emotion they seek to avoid in the audience. For example, Ajam can be used for a poem recited at a wedding, and Saba can be for a poem recited at a funeral. One can also switch between scales for different parts of a (long) poem, depending on what one wants to communicate.


For learners, learning to recognize the scales is essential to learning to recite or sing them. However, there is a large barrier to entry. It is very difficult to learn to recognize them confidently without the help of a professional. This makes them very inaccessible to learn. For this reason, I have created a classifier that can detect the scale of an audio segment. This project serves as a precursor to a learning tool. In the long term, this project can sprout into a full-fledged learning tool can become vital for students to learn the art of recital and singing. 


For this project, I have created a completely original dataset of 30.5 hours. This is the largest dataset of its kind for the 8 Arabic scales. By completing this project, I lay the foundations for a complete pipeline that can classify long audio segments.


## Previous Literature 


There have been 3 academic papers ([Shahriar](https://ieeexplore.ieee.org/document/9496604), [Omari](https://figshare.com/articles/journal_contribution/Maqam_Classification_of_Quranic_Recitations_using_Deep_Learning/24131781?file=42335634), and [Alaydrus](https://browser-cdn.ysoa.org.uk/volumes/Vol101No21/34Vol101No21.pdf)) written in total on the topic of classifying maqams, done on 2 datasets (which are not publicly available). The first dataset is 6 hours, and the second is 30 hours long. The focus on these papers is on classifying 30-60 second audio clips. Overall, accuracies of 80-95% have been achieved.


## Overview


My original dataset contains a matching dataset of 30-60 second clips & JSON files with their corresponding scale labels. The creation of this dataset is outside the scope of this project, but will be summarized below.
From each clip, I extracted 17 features (see below). I used an artificial neural network to train an algorithm to classify each 30-60 second clip. The trained network is then used for a segment classifier that is capable of assigning labels to segments with unlimited length.


![Alt Text](https://raw.githubusercontent.com/mohssenk/Arabic-Melodies-Project/refs/heads/master/images_for_introduction/NN_diagram.png)


## Creation of Dataset


The dataset was created using a publicly available [playlist](https://www.youtube.com/playlist?list=PL97dVNc_FCMVlCaSqgYdictI0qHWeLG7D) by a professional reciter called Mustafa Shukeir, who uploaded 30.5 hours of himself doing a Quranic recitation while alternating between the 8 scales. This is an educational playlist which always has the scale written on screen. Using an automated pipeline built with Python and relying on the Tesseract Optical Character Recognition (OCR) library, I chopped up each video in the playlist into segments (317 total segments across the playlist), with each segment being the part of the video where he used a particular scale. Because he would alternate between scales during a video, each video becomes multiple segments. 


Each segment was further chopped into 30-60 second audio clips (3814 total clips) and corresponding labels (stored in JSON). Typically, all the clips extracted from a segment would be 30 seconds, except the final clip which would be between 30-60 seconds. These clips are used for training the neural network. Each label is a json file that records the timestamp from the video it corresponds to, the scale, and other features that were provided by the playlist but are beyond the scope of this project. (These extra recorded features are the pitch and the branch, which is a subscale). Much of the information recorded in the json is in Arabic, and is translated to English during processing.


The scripts used for the creation of the dataset are not included within this project, but may be added in future updates.


![Alt Text](https://raw.githubusercontent.com/mohssenk/Arabic-Melodies-Project/refs/heads/master/images_for_introduction/data_dist.png)


## Features


The following 17 features were extracted from the audio clips to use at inputs:

  - 12 chroma features 
  - Root mean square
  - Zero crossing rate
  - Spectral centroid
  - Spectral bandwidth
  - Spectral rolloff


## Neural Network Architecture


The architecture of the neural network is shown below. It consists of 6 layers and was inspired by the architectures in the previous literature. A dropout of 0.4 and 0.25 were used in the first 2 layers and middle 2 layers, respectively. Batch Normalization was used in all layers. The batch size used was 64 and the learning rate was 0.0001. LeakyRelu was used in the hidden layers, and softmax was used to determine the output.


![Alt Text](https://raw.githubusercontent.com/mohssenk/Arabic-Melodies-Project/refs/heads/master/images_for_introduction/NN_architecture.png)

![Alt Text](https://raw.githubusercontent.com/mohssenk/Arabic-Melodies-Project/refs/heads/master/images_for_introduction/model_eval.png)


## Results


The accuracy on the test data was 84.3%. Across the 8 scales, the recall hovered between 70-96%. A more detailed breakdown can be seen in the figures below.


**Overall Performance Metrics**


| Metric      | Value |
|------------|-------|
| **Accuracy**  | 0.843 |
| **Precision** | 0.847 |
| **Recall**    | 0.843 |
| **F1-Score**  | 0.843 |

---

**Per-Class Recall Scores**

| Scale   | Recall |
|---------|--------|
| **Bayat**   | 0.70  |
| **Hejaz**   | 0.96  |
| **Rast**    | 0.76  |
| **Seekah**  | 0.87  |
| **Saba**    | 0.90  |
| **Ajam**    | 0.84  |
| **Kurd**    | 0.86  |
| **Nahawand** | 0.89  |


![Alt Text](https://raw.githubusercontent.com/mohssenk/Arabic-Melodies-Project/refs/heads/master/images_for_introduction/confusion_matrix.png)


## Segment Classifier


Finally, I created a segment classifier that can take in a long segment and detect the scale. It does so by cutting the segment into 30 second audio clips, classifying the clips, and then selecting a final scale through majority voting. 

![Alt Text](https://raw.githubusercontent.com/mohssenk/Arabic-Melodies-Project/refs/heads/master/images_for_introduction/classifier_diagram.png)


## Future Plans


Currently, the dataset is only 1 reciter, so to be generalizable it needs a much more diverse dataset. I plan to expand the dataset heavily by adding uploaded youtube videos with the scale listed in the description or title. With my unique contribution of an audio classifier, an application will be created in the future that can classify any clip uploaded by a learner that wants to confidently know the scale of a reciter they listen to. 

