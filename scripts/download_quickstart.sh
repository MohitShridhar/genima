mkdir ckpts
cd ckpts
wget https://github.com/MohitShridhar/genima/releases/download/1.0.0/25_tasks.zip
unzip 25_tasks.zip
rm 25_tasks.zip

sed -i 's/tiger/robobase/g' 25_tasks/controller_act/config.yaml