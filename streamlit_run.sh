cd w210-streamlit
tmux new -s StreamSession
streamlit run 210_Home.py
# ctrl + b
# d
# tmux attach -t StreamSession
# ps aux | grep streamlit | sed /ubuntu/
ps aux | grep streamlit | grep -Po 'ubuntu\s+\d+' | grep -Po '\d+'
tmux kill-session 