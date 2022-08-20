# Hard-label-Black-box-Attack
Team Members:
* Parth Shettiwar
* Rohit Sunchu
* Satvik Mashkaria

Done in the course CS269- Adverserial Learning

Major Contributions in this project:
* Inspired from Schott et al. [2019], which was done in white box setting, we propose
a homotopy based algorithm, SparseAPG to perform hard-label black-box attack which
minimizes the l0 norm.

* Our method also bounds l-infinity norm while solving the objective, unlike the existing methods
like Vo et al. [2022]. Also, our attack produces human imperceptible images at every
iteration, unlike greedy methods like Schott et al. [2018].

* We shows results of our attack on CIFAR-10 images with trained ResNet-18 model (?), and
compare the performance with the existing attacks in our setup.
