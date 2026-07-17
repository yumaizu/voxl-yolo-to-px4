# PX4 YOLO Detection Response System

A ROS 2 and MAVSDK-based Python application that receives YOLO
object-detection results from a ModalAI VOXL system and executes a
configurable action on a PX4 flight controller when a specified object
is detected.

The current implementation supports:

-   Receiving YOLO inference results through a ROS 2 topic
-   Connecting to PX4 using MAVSDK
-   Switching PX4 to Hold mode when a detection is triggered
-   Commanding PX4 to Land when a detection is triggered
-   Extensible action callbacks for future behaviors

## Architecture

``` text
┌──────────────────────────────┐
│        YOLO / TFLite         │
│      Object Detection        │
└──────────────┬───────────────┘
               │
               │ ROS 2
               │ /yolo_tflite_data
               ▼
┌──────────────────────────────┐
│        YOLOReceiver          │
│                              │
│  Receives Aidetection msgs   │
│  Checks object + confidence  │
└──────────────┬───────────────┘
               │
               │ Action callback
               ▼
┌──────────────────────────────┐
│       PX4Connector           │
│                              │
│       MAVSDK / PX4           │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│             PX4              │
│                              │
│     Hold / Land / Future     │
│          Actions             │
└──────────────────────────────┘
```

## Project Structure

``` text
.
├── main.py
├── PX4Connector.py
├── YOLOReceiver.py
└── README.md
```

### `main.py`

The application entry point.

Responsibilities:

-   Stores application configuration
-   Starts the asyncio event loop
-   Initializes ROS 2
-   Creates the `YOLOReceiver`
-   Creates the `PX4Connector`
-   Selects the action to execute after detection
-   Connects the YOLO receiver to the PX4 action callback
-   Starts and shuts down the application

### `YOLOReceiver.py`

Receives inference results from the YOLO pipeline.

Responsibilities:

-   Subscribes to the configured ROS 2 topic
-   Receives `voxl_msgs/Aidetection` messages
-   Logs detection information
-   Checks the detected class and confidence
-   Prevents repeated triggering after the first valid detection
-   Executes the configured asynchronous action callback

The receiver itself does not perform object detection. The actual
inference is performed by the YOLO/TFLite pipeline.

### `PX4Connector.py`

Handles communication with the PX4 flight controller through MAVSDK.

Responsibilities:

-   Connects to PX4
-   Provides flight-control commands
-   Handles MAVSDK errors
-   Provides asynchronous methods that can be used as callbacks

Currently available actions:

-   `set_hold_mode()`
-   `set_land_mode()`

## Configuration

Configuration is located at the top of `main.py`.

### PX4 Connection Address

``` python
PX4_ADDRESS = 'udp://:14551'
```

The address used to connect to PX4 through MAVSDK.

Supported formats include:

``` text
Serial:
serial:///path/to/serial/dev[:baudrate]

UDP input:
udpin://bind_host:bind_port

UDP output:
udpout://dest_host:dest_port

TCP input:
tcpin://bind_host:bind_port

TCP output:
tcpout://dest_host:dest_port
```

For example:

``` python
PX4_ADDRESS = 'udp://:14551'
```

### YOLO Detection Topic

``` python
YOLO_DETECTION_TOPIC = '/yolo_tflite_data'
```

The ROS 2 topic from which detection messages are received.

The expected message type is:

``` text
voxl_msgs/Aidetection
```

### Detection Action

``` python
DETECTION_ACTION = 'hold'
```

This determines what the PX4 vehicle should do when a valid detection is
received.

Available actions:

``` python
DETECTION_ACTION = 'hold'
```

or:

``` python
DETECTION_ACTION = 'land'
```

## Detection Behavior

When a message is received, the receiver checks:

``` python
msg.class_name == 'person'
```

and:

``` python
msg.class_confidence >= 0.6
```

The action is only triggered when both conditions are true.

The receiver also maintains a trigger state:

``` python
self.hold_triggered = False
```

After a valid detection:

``` python
self.hold_triggered = True
```

This prevents the action from being repeatedly executed for every
subsequent detection message.

> Note: The variable name `hold_triggered` can be renamed to something
> more general such as `action_triggered` if actions other than Hold are
> used. This would be a recommended future improvement.

## Action Callback System

The action is selected using an action registry in `main.py`:

``` python
actions = {

    'hold': px4_connector.set_hold_mode,

    'land': px4_connector.set_land_mode,

}
```

The configured action is then assigned to the receiver:

``` python
yolo_receiver.action_callback = (
    actions[DETECTION_ACTION]
)
```

When a valid detection occurs:

``` text
Detection
    │
    ▼
YOLOReceiver
    │
    ▼
action_callback()
    │
    ▼
Selected PX4 action
    │
    ├── set_hold_mode()
    │
    └── set_land_mode()
```

This design allows the YOLO receiver to remain independent of the
specific PX4 action being executed.

## Adding a New Action

To add a new PX4 action, first add an asynchronous method to
`PX4Connector.py`.

For example:

``` python
async def set_return_to_launch_mode(self):

    try:

        self.logger.warn(
            'Sending Return-to-Launch command to PX4'
        )

        await self.drone.action.return_to_launch()

        self.logger.warn(
            'PX4 Return-to-Launch command successfully sent'
        )

    except ActionError as e:

        self.logger.error(
            'Failed to send Return-to-Launch command: {}'.format(e)
        )

    except Exception as e:

        self.logger.error(
            'Unexpected MAVSDK error: {}'.format(e)
        )
```

Then add the method to the action registry in `main.py`:

``` python
actions = {

    'hold': px4_connector.set_hold_mode,

    'land': px4_connector.set_land_mode,

    'rtl': px4_connector.set_return_to_launch_mode,

}
```

Finally, select it in the configuration:

``` python
DETECTION_ACTION = 'rtl'
```

No changes to `YOLOReceiver.py` are required.

## Asynchronous Architecture

The application uses two event systems:

### ROS 2

ROS 2 handles YOLO detection messages:

``` text
ROS 2
  │
  ▼
YOLOReceiver.detection_callback()
```

### asyncio

MAVSDK uses asyncio for PX4 communication.

The asyncio event loop runs in a separate thread:

``` text
Main Thread
│
└── ROS 2 / rclpy.spin()

Asyncio Thread
│
└── MAVSDK / PX4 communication
```

When a detection occurs, the asynchronous PX4 action is safely scheduled
on the MAVSDK event loop:

``` python
asyncio.run_coroutine_threadsafe(
    self.action_callback(),
    self.loop
)
```

This allows the ROS 2 callback and MAVSDK asyncio code to operate
together without blocking each other.

## Requirements

The system requires:

-   Python 3
-   ROS 2
-   `rclpy`
-   `voxl_msgs`
-   `mavsdk`
-   A running YOLO/TFLite inference pipeline publishing
    `voxl_msgs/Aidetection`
-   A PX4 flight controller accessible through MAVSDK

## Running

Make sure the following are running:

1.  PX4
2.  MAVSDK connection endpoint
3.  YOLO/TFLite inference pipeline
4.  ROS 2 environment

Then run:

``` bash
python3 main.py
```

The application will:

1.  Start the asyncio event loop
2.  Initialize ROS 2
3.  Subscribe to the YOLO detection topic
4.  Connect to PX4
5.  Wait for YOLO detections
6.  Execute the configured action when a valid detection is received

## Example

With:

``` python
PX4_ADDRESS = 'udp://:14551'

YOLO_DETECTION_TOPIC = '/yolo_tflite_data'

DETECTION_ACTION = 'land'
```

A detection such as:

``` text
Class: person
Confidence: 0.87
```

will result in:

``` text
PERSON DETECTED - executing configured action
Sending Land command to PX4
PX4 Land command successfully sent
```

## Future Improvements

Potential future improvements include:

-   Rename `hold_triggered` to `action_triggered`
-   Make the detection class configurable
-   Make the confidence threshold configurable
-   Support multiple detection classes
-   Support different actions for different classes
-   Add a cooldown period between actions
-   Add a configurable action sequence
-   Add additional PX4 actions such as:
    -   Return to Launch
    -   Disarm
    -   Pause mission
    -   Start mission
    -   Change flight mode
-   Add graceful cancellation of active MAVSDK commands
-   Add configuration through a YAML file or ROS 2 parameters
-   Add safety checks before executing flight actions
