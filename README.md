# APnP

This repository contains code and resources related to the paper [AP$n$P: A Less-constrained P$n$P Solver for Pose Estimation with Unknown Anisotropic Scaling or Focal Lengths]() by Jiaxin Wei, Stefan Leutenegger, and Laurent Kneip.

![llustration of the two practical cases in camera pose estimation with relaxed constraints.](images/teaser.jpg)


## Abstract

> Perspective-$n$-Point (P$n$P) stands as a fundamental algorithm for pose estimation in various applications. In this paper, we present a new approach to the P$n$P problem with relaxed constraints, eliminating the need for precise 3D coordinates or complete calibration data. We refer to it as AP$n$P due to its ability to handle unknown anisotropic scaling factors of 3D coordinates or alternatively two distinct focal lengths in addition to the conventional rigid pose. Through algebraic manipulations and a novel parametrization, both cases are brought into similar forms that distinguish themselves primarily by the order of a rotation and an anisotropic scaling operation. AP$n$P furthermore brings down both cases to an identical polynomial problem, which is solved using the Gr\"obner basis approach. Experimental results on both simulated and real datasets demonstrate the effectiveness of AP$n$P, providing a more flexible and practical solution to several pose estimation tasks.