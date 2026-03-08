# Guide: Running Local Code on ARCC GPUs (MedicineBow)

To run your GPU-heavy code (`copalirag` and `neo4j` ingestion) on University of Wyoming's ARCC HPC clusters (like MedicineBow), you need to transfer your code, set up your environment, and request a GPU node using SLURM. 

By default, large workloads and project code should be stored in your project's `gscratch` directory rather than your home directory to avoid quota limits.

## Step 1: Connect to the UW Network
You must be connected to the UW network. If you are off-campus, connect to **WyoSecure VPN**.

## Step 2: Transfer Your Code to ARCC (`gscratch`)
Since you want to store your code at `/gscratch/nsubedi1/`, use the `rsync` command from your Mac terminal to push the files directly to the cluster.

> **Note:** Run this command from your Mac terminal (do NOT ssh into ARCC yet).

```bash
# Run this on your Mac:
rsync -avz --exclude '.git' --exclude '__pycache__' ~/Desktop/wydot_cloud/ nsubedi1@medicinebow.arcc.uwyo.edu:/gscratch/nsubedi1/wydot_cloud/
```

*What this does:*
* Transfers everything from `~/Desktop/wydot_cloud/` to `/gscratch/nsubedi1/wydot_cloud/` on MedicineBow.
* Ignores the `.git` directory and Python caches to save time.
* If you make changes on your Mac later, just run this command again to quickly sync the updates!

## Step 3: SSH into the Cluster
Once the transfer completes, SSH into MedicineBow.

```bash
# SSH into the cluster (replace nsubedi1 if your local SSH username differs)
ssh nsubedi1@medicinebow.arcc.uwyo.edu
```
*(Enter your UW password and complete 2FA if prompted).*

## Step 4: Set Up Your Conda Environment (`wyoenv`)
Navigate to your newly transferred project folder and set up `wyoenv`. ARCC provides miniconda as a module.

```bash
# 1. Go to your project directory
cd /gscratch/nsubedi1/wydot_cloud

# 2. Load the miniconda module
module load miniconda3

# 3. Create the environment (only needed the first time)
conda create -n wyoenv python=3.11 -y
conda activate wyoenv

# 4. Install dependencies
pip install -r requirements.txt
```

## Step 5: Request an Interactive GPU Session
Do **NOT** run your heavy python code on the login node. You must request a compute node with a GPU using SLURM (`salloc`). 

Request a node with 1 GPU for 4 hours interactively:

```bash
salloc --account=<your_project_account> --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=04:00:00 --gpus=1
```
*(Replace `<your_project_account>` with your actual ARCC PI project name).*

Once the allocation is granted, your terminal prompt will change to a compute node (e.g., `md001`).

## Step 6: Run Your Code on the GPU Node
Now that you are on the compute node, load your environment and run the code!

```bash
# 1. Be sure you are in the correct directory
cd /gscratch/nsubedi1/wydot_cloud

# 2. Ensure modules and conda are loaded on this new node
module load miniconda3
conda activate wyoenv

# 3. Run the ingestion scripts
# For Copali visual ingestion:
python copalirag/visual_retriever.py 

# Or for Neo4j graph building:
python neo4j/ingestneo4j.py
```

---

## Alternative: Submitting a Background Job (`sbatch`)
If ingestion takes many hours, it's better to submit it as a background job so it continues running even if you close your laptop.

**1. Create a file called `run_ingest.sh` in your project folder:**
```bash
#!/bin/bash
#SBATCH --job-name=wydot_gpu_job
#SBATCH --account=<your_project_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --output=/gscratch/nsubedi1/wydot_cloud/ingest_%j.log

# Go to your scratch directory
cd /gscratch/nsubedi1/wydot_cloud

# Load environment
module load miniconda3
source activate wyoenv

# Run the python script
python copalirag/visual_retriever.py
```

**2. Submit it to the cluster:**
```bash
sbatch run_ingest.sh
```

**3. Check on it:**
```bash
squeue -u nsubedi1
```
You can view the output by reading the generated `ingest_XXXX.log` file in the gscratch folder.
