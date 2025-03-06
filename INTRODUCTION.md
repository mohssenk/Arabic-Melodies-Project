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
These 8 scales aren’t just used for musical instruments, but for singing and reading poems melodiously. Melodiously orating long, well-written poems is a very large part of Arabic culture, and the 8 scales come heavily in handy. One can choose a scale to use while orating their poem, depending on the emotion they seek to avoid in the audience. For example, Ajam can be used for a poem recited at a wedding, and Saba can be for a poem recited at a funeral. One can also switch between scales for different parts of a (long) poem, depending on what one wants to communicate.
For learners, learning to recognize the scales is essential to learning to recite or sing them. However, there is a large barrier to entry. It is very difficult to learn to recognize them confidently without the help of a professional. This makes it very inaccessible to learn. For this reason, I have created a classifier that can detect the scale of an audio segment. This project serves as  a precursor to a learning tool. In the long term, this learning tool can become vital for students to learn the art of recital and singing. 
For this project, I have created a completely original dataset of 30.5 hours. This is the largest dataset of its kind for the 8 Arabic scales. By completing this project, I lay the foundations for a complete pipeline that can classify long audio segments.

## Previous Literature 

There have been 3 academic papers ([Shahriar](https://ieeexplore.ieee.org/document/9496604), [Omari](https://figshare.com/articles/journal_contribution/Maqam_Classification_of_Quranic_Recitations_using_Deep_Learning/24131781?file=42335634), and [Alaydrus](https://browser-cdn.ysoa.org.uk/volumes/Vol101No21/34Vol101No21.pdf)) written in total on this project, done on 2 datasets (which are not publicly available). The first dataset is 6 hours, and the second is 30 hours long. The focus on these papers is on classifying 30-60 second audio clips. Overall, accuracies of 80-95% have been achieved.

## Overview

My original dataset contains a matching dataset of 30-60 second clips & JSON files with their corresponding scale labels. The creation of this dataset is outside the scope of this project, but will be summarized below.
From each clip, I extracted 17 features (see below). I used an artificial neural network to train an algorithm to classify each 30-60 second clip. The trained network is then used for a segment classifier that is capable of assigning labels to segments with unlimited length.

![Alt Text](https://raw.githubusercontent.com/mohssenk/Arabic-Melodies-Project/refs/heads/master/images_for_introduction/NN_diagram.png)
