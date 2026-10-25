# Isel Rail Controller

This folder drives the three-axis isel rail of the obstacle experiment by hand: jogging, zeroing and measuring each
axis.

## Commands

```
pip install -r requirements.txt
```

| Command | What it does |
| :--- | :--- |
| `python terminal_controller_3axis.py` | Interactive terminal; `help` lists the commands. |
| `python controller_connection_test.py` | Releases the brakes and moves one axis as a test. |
| `python loopback_test.py` | Checks the serial cable alone. |

Positions are motor steps; `measure <axis> <mm>` gives the true factor $k = \text{steps}/\text{mm}$ against a ruler.

## How to Measure

1. Connect the controller and switch the rail on: `python terminal_controller_3axis.py`
2. Drive to the start position and zero every axis: `sethome 0`, `sethome 1`, `sethome 2`
3. Measure the steps per millimetre of an axis: `measure 1 100`

## Folder Structure

```
isel/
├── terminal_controller_3axis.py   interactive terminal for the three axes
├── controller_connection_test.py  brake release and a test move
└── loopback_test.py               serial cable check
```
