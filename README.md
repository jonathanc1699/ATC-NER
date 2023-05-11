# ATC-NER
Repo for internship documents/code

How to train NER model:
Train all ner model with input dataset which is augmented and in csv format:

    {
        "training_type": "bulk",
        "models": [
            "distilbert",
            "bert",
            "roberta"
        ],
        "csv_input_type": true,
        "input_full_csv": "./data_csv/ATC_Data_full.csv",
        "input_train_csv": "./data_csv/ATC_Data_train.csv",
        "input_test_csv": "./data_csv/ATC_Data_test.csv",
    }

Run 'ner_model.py' in Gekko - NTU Server
Install MobaXterm on Windows to a linux server - link

Install WinSCP

Open mobaxterm

Open session and click ssh

 ssh -X [NTU_username]@gekko.hpc.ntu.edu.sg
 cd Desktop
 mkdir scratch/
Transfer ner_model.py, and data folder into the scratch folder in gekko environmeny

Copy the pbs file from the gekko folder of this github into gekko environment

Run script on server

 qsub gmb.pbs 
To check on status of script

 qstat -u $USER



Instructions for Pipeline Reproduction
Setting up enviroment ( To be done during initial setup of repo - only once )
i. Install conda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

ii. Create virtual env (https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

iii. Install fst and thrax

conda install -c conda-forge thrax
conda install -c conda-forge openfst 

Activating virtual env Eg :
conda activate conda_env_name
Install requirements
pip install -r requirements.txt
( Here conda_env_name is same as the enviroment created earlier in Step 1-ii. )


Making sure all directories are setup properly
main_directory
input
exp_id_name
logs
output
exp_id_name
json_env
Setting up env parameters in env file.
Go to env/env_name.json and set it up according to requirements

( Sample env.json is given at the start of readme documentation for reference )


Running the script Make sure you are inside the main folder and run test_5.sh with env_name ( where env_name is the name of the json file inside env folder which has all the env variables specified )
./textgen_script.sh json_env/env1.json

Input Enviroment Variables
Sample arguments for working with ATC dataset for text generation for Language Modelling and NER tasks.


{
	"input_folder" : "input/atc",
	"entry_grm" : "atc.grm",
	"far_file" : "atc.far",
	"exp_id" : "atc_1", //any name
	"exp_description": "test for 1000 atc sentences", //any description
	"num_of_output_sentence": 1000, //any amount
	"generate_bilou" : "FALSE", //set to True for NER augmentation
	"output_folder" : "output/", /output folder name
	"txt_file_name" : "test1000.txt", /output file name
	"log_file" : "test.log"
}
