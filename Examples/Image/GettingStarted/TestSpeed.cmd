del /s /q output > nul
echo . | time
cntk configFile=01_OneHidden.cntk
echo . | time
python x:\Repos\CNTK\bindings\python\examples\MNIST\SimpleMNIST.py
echo . | time