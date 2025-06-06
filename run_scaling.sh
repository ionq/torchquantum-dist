#weak: increase job size and resources
N=(1 2)
B=(16 16)
Q=(3 4)
for i in {0..1}; do
    torchrun --nproc-per-node=${N[i]} scaling.py ${B[i]} ${Q[i]} ${N[i]}
done


#strong: constant job size while increasing resources
N=(1 2)
B=(16 16)
Q=(4 4)
for i in {0..1}; do
    torchrun --nproc-per-node=${N[i]} scaling.py ${B[i]} ${Q[i]} ${N[i]}
done