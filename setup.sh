#!/bin/bash

tmp_dir='tmp123123'
git clone https://github.com/MultiAgentLearning/playground.git ${tmp_dir}
cd ${tmp_dir}

echo -e "\nInstalling the 'pommerman MultiAgentLearning' library.."
echo -e " NB: it installs using your default pip if you want to install with conda
           or other pip do it manually or set your desired virtual environment!\n"



pip install -U .
cd ../
rm -rf ${tmp_dir}

