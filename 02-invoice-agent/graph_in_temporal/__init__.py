"""Build C1: the whole LangGraph investigation inside one Temporal activity. LangGraph runs the loop and
saves a checkpoint after every node; Temporal restarts the activity after a crash and waits for the human."""
