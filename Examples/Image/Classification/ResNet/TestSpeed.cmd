del /s /q output > nul
echo . | time
cntk configFile=ResNet20_CIFAR10.cntk
echo . | time
python x:\Repos\CNTK\bindings\python\examples\CifarResNet\CifarResNet.py
echo . | time