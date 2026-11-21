# Hub setup: a master copy, and one copy per user

For the tutorial server (The Littlest JupyterHub, users `jupyter-<name>`, group `jupyterhub-users`).

| Where | What | Who can write |
|---|---|---|
| `/home/smc-tutorial/master/mosaic`, `.../shasta` | the master copy of each tutorial | admins only |
| `/home/smc-tutorial/scores/mosaic` | the leaderboard's score files | every user (one file each) |
| `~/mosaic`, `~/shasta` | a user's own copy, made when their Jupyter server starts | that user |

## One-time setup

Put `setup.sh` and `smc-sync` in one folder on the server and, as an admin:

```bash
sudo bash setup.sh
```

It backs up the current shared `mosaic` and `shasta` folders to `/home/smc-tutorial/_backup`, makes them the masters (without run leftovers or saved notebook output), creates the scores folder,
removes the group's write access to the top folder, installs `/usr/local/bin/smc-sync` (linked from `/opt/tljh/user/bin`, which is on the users' `PATH`), and adds a hook to `/opt/tljh/user/etc/jupyter/jupyter_server_config.py` that runs `smc-sync` whenever a
user's server starts. Running it again is safe.

## Changing the masters

Edit the master, never a user's copy. For example, after copying new files to the server:

```bash
sudo rsync -a --chown=root:root --exclude='.ipynb_checkpoints' --exclude='__pycache__' ./notebooks/ /home/smc-tutorial/master/mosaic/
sudo chmod -R u=rwX,go=rX /home/smc-tutorial/master
```

Check that no run leftovers (`*.png`, `*.xdf`, `scores/`) are in the master: every user gets what is there.

At a user's next login, `smc-sync` adds any new file and refreshes the helper `.py` files (`live_play.py`, `lsl_tools.py`, `world_view.py`, `leaderboard.py`), which users should not edit. Their notebook and
`config.yaml` are never overwritten. A user who wants the new notebook runs `smc-sync --reset mosaic` in a terminal: the old folder is kept as `mosaic.backup-<time>`, and the new copy appears.

## Check it

```bash
sudo -u jupyter-bennett /usr/local/bin/smc-sync && ls /home/jupyter-bennett     # mosaic and shasta appear
ls -l /home/smc-tutorial/master/mosaic                                           # owned by root
sudo -u jupyter-bennett touch /home/smc-tutorial/master/mosaic/x                 # must fail: permission denied
```

Then log in as a test user: the folders should be there before you open anything. If they are not, the hook was not read: run `smc-sync` once by hand in that user's terminal and tell whoever
maintains the server.
