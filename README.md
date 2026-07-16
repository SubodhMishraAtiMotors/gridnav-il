cat > README.md <<'EOF'
# gridnav-il

Imitation learning for grid-world robot navigation.

Pipeline:
1. Generate random grid worlds with rectangular obstacles.
2. Compute traversal cost and Dijkstra cost-to-go.
3. Track planned path using Pure Pursuit expert.
4. Log robot-frame local cost observations and expert actions.
5. Train CNN policy to imitate expert actions.
6. Evaluate CNN policy in closed-loop against Pure Pursuit.

## Setup

```bash
conda create -n gridnav-il python=3.10 -y
conda activate gridnav-il
pip install numpy scipy matplotlib tqdm pyyaml scikit-learn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
