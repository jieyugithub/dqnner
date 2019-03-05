## Document-level NER with Reinforcement Learning

This code project is implemented based on Karthik Narasimhan's work (https://github.com/karthikncode/DeepRL-InformationExtraction).

### Installation
You will need to install [Torch](http://torch.ch/docs/getting-started.html) and the  python packages in `requirements.txt`.  

You will also need to install the Lua dev library `liblua` (`sudo apt-get install liblua5.2`) and the [signal](https://github.com/LuaDist/lua-signal) package for Torch to deal with SIGPIPE issues in Linux.
(You may need to uninstall the [signal-fft](https://github.com/soumith/torch-signal) package or rename it to avoid conflicts.)

### Running the code
  * First run the server, for example:
    python ~/dqnner/Code/code/server_ner.py

  * In a separate terminal/tab, then run the agent:
    cd ~/dqnner/Code/code/dqn
    ./run_cpu.sh AKWS1News_E Bagging 49021 20190113-18-T7-Pse-Pctx1-20-20-PartTag1-EO1-DEMO 423 9925 NER_SIA 20 20 0.8

  * NOTICE:
    Make sure that the following files are in the folder '/mnt/hgfs/VMWareShare/20190113-18-T7-Pse-Pctx1-20-20-PartTag1-EO1-DEMO/AKWS1News_E/Bagging'.
    - test.gold.json
    - test.keys.json
    - test.map.json
    - train.gold.json
    - train.keys.json
    - train.map.json
    These file can be found in our project: dqnner/Data.

  * If you have any question, don't hesitate to contact lutingming@163.com.

### Acknowledgements
  * [Karthik Narasimhan's DQN4IE codebase](https://github.com/karthikncode/DeepRL-InformationExtraction)
  * [Deepmind's DQN codebase](https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner)

