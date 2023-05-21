Hi Tushaar,

In case you are reading this it is because I am no longer one of yours... It is a sad moment to have to say goodbye and assume my life as just another Computer Sciencist in the industry. Go on, be bold and make me pround in the Academia world. I am 100% confident that you are much better than all our peers, believe in youself and don't be afraid of failing, be brave and ask the hard questions.

We had a great time together, and hopefully we were able to build something very cool and that hopefully will make some buss.


About the server there are three main parts you gotta understand:

1. **Slurm commands**

In your .sh file it has to have the following content:

example.sh

#!/bin/sh
#Especifications
echo "Hi Tushaar, you are an asshole"
python3 example.py
python3 anotherExample.py

Important to use python3, otherwise it won't work!

The #Especification parts there is a lot you can do, but so far the ones I have found most useful were:
#SBATCH -N 2 -> _Here you will set up the number of nodes you want to use_
#SBATCH --mem=64000 -> _Here you will set up the amount of memory you want to use (in MB). Don't worry, if ask more tham the GPU you are assigned for it will crash instantly._
#SBATCH --time= 05:00:00 -> _Here I am setting it to run for 5 hours. The default is 5. There is probably a way to do remove time limits, but you gotta aske Rich, I don't know how to do it._

Honestly, all the tests I ran I used only those, but there a couple more. The one that would be useful is choosing the GPU you want.

In order to run your example.sh the best is to just do:

sbatch example.sh

Although it is possible to run with the combination of some other commands, such as srun followed by some command to assign that process to a given node, I don't how to do it, and I don't recommend, because if you do it wrong you will be running in the head node and you get banned.

As soon as you do sbatch you receive you process ID and a slurm<Process ID>.out file will be generated. I don't know the proper definition of this file, but I see it as a logger file. If anything crashes in runtime, this is the place where you will see it. So just vim slurm<Process ID>.out every now and then. 

And the a handfull of useful stuff is:
* squeue: allows you to see the processes that are running and for how long they have been running (so you can see if it is all fine with your process)
* scancel: will allow to kill your process
* scontrol show list: you will be able to see the specifications of the available units.

2. Conda and packages

I know you don't like Conda, but I don't cara, get use to it, and don't bitch about it. 

This is somewhat misterious thing that I don't really understand on the server. Everything you save in your home depository you will be there permanently (but not backed up), however the configurations you change are reset every single time you have to reconnect.

So the best is to dowload the conda installer once with:

wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
**Pay Attention because it has this version of conda, given the python version of the server**
bash Anaconda3-5.2.0-Linux-x86_64.sh
Just confirm everything...
source ~/.bashrc
conda create --name togepi


And then, the final aspect that is important, in order to install the packages you should do:

pip3 install --user <package>

Almost all set...

3. Git hub

The git hub is also annoying because you get yous access revoked every time you disconnect, I honestly don't really know the reason. Rich said that there a solution for that, but in that case you would give access to any one in the server with access to your directory. So to avoid any misfortunate incidents I decided not to do it. The solution is create a new SSH everytime you reconnect:


ssh-keygen -t ed25519 -C "your_email@example.com"
Enter
Enter
Enter
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519

Copy the key, add it to your gitHub account and you are all set! 

run this: ssh -T git@github.com

To make sure you connected!


**Final remarks:**

It is kinda of annoying to this everytime, so the best solution is to use tmux to prevent your session from disconnecting.

PS.: FileZilla work the same way it would in any other server.

Best,

Lucas

PS.: I hope you are eating well... healthy food and solid meals.


































