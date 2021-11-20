import sys
import argparse
import time
import datetime
import platform
import json
import csv
import numpy as np
import pandas as pd
from os.path import exists
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams
from brainflow.exit_codes import *

# TODO HANDLE SERIAL PORTS: COM3 VS /DEV/TTY/ VS /DEV/CU
# https://brainflow.readthedocs.io/en/stable/SupportedBoards.html

BRAINWAVES = ['delta', 'theta', 'alpha', 'beta', 'gamma']
ROUNDING_DECIMAL_PLACE = 3
TIMEOUT = 60


def main():
    arguments = parse_script_arguments(sys.argv[1:])
    eeg_num = arguments['eeg']
    mac = arguments['mac']

    writer = prepare_csv(arguments['song'], arguments['user'])

    eeg_params = get_eeg_params(eeg_num, mac)
    board = create_eeg_board_object(eeg_params)
    stream_eeg_data(board, eeg_params, writer)


def parse_script_arguments(args):
    '''Input: system args. Output: Variables as json. Defaults EEG to ganglion unless user specified otherwise.'''
    eeg = 'Ganglion'
    user = ''
    song = ''
    mac = ''

    for i in range(len(args)):
        if (args[i] == "-eeg" or args[i] == "--eeg") and i+1 < len(args):
            eeg = args[i+1]
        if (args[i] == "-u" or args[i] == "--user") and i+1 < len(args):
            user = args[i+1]
        if (args[i] == "-s" or args[i] == "--song") and i+1 < len(args):
            song = args[i+1]
        if (args[i] == "-m" or args[i] == "--mac") and i+1 < len(args):
            mac = args[i+1]

    print_debug(str("userheadset: " + eeg))
    return {"eeg": eeg, "mac": mac, "user": user, "song": song}


def prepare_csv(song, user):
    filename = get_unique_filename(song, user)
    f = open(filename, 'w')
    writer = csv.writer(f)
    headers = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'note', 'chord']
    writer.writerow(headers)
    return writer


def get_unique_filename(song, user):
    current_time = datetime.datetime.now()
    cur_date = current_time.month + current_time.day + current_time.year
    cur_time = current_time.hour + current_time.minute + current_time.second
    file_extension = ".csv"
    filename = song + "_" + user + "_eeg_" + cur_date + cur_time + file_extension

    index = 0
    while (exists(filename)):
        index = index + 1
        filename = song + "_" + user + "_eeg_" + \
            cur_date + cur_time + index + file_extension

    return filename


def get_eeg_params(eeg_name, mac_address):
    """ Configures EEG Headset board specifications for EEG headsets based off of Brainflow Documentation """
    # There might be differences for each of the boards, tried the best we could since we only had one type of EEG board!
    # Here's for more info: https://brainflow.readthedocs.io/en/stable/SupportedBoards.html

    # 1. Build the variables
    board_id = 1  # Default is the Ganglion
    serial_port = ''
    ip_address = ''
    ip_port = 0
    ip_protocol = 0
    other_info = ''
    serial_number = ''
    file = ''
    streamer_params = ''
    # mac_address is default to '' when inputting the EEG data into the terminal :)

    # OpenBCI Devices:
    if (eeg_name == 'Synthetic'):
        board_id = -1
    if (eeg_name == 'Cyton'):
        board_id = 0
        serial_port = 'COM3'
    if (eeg_name == 'Ganglion'):
        board_id = 1
        serial_port = getSerialPortBasedOffOfOS()
    if (eeg_name == 'CytonDaisy'):
        board_id = 2
        serial_port = 'COM3'
    if (eeg_name == 'CytonWIFI'):
        board_id = 4
    if (eeg_name == 'GanglionWIFI'):
        board_id = 5
    if (eeg_name == 'CytonDaisyWIFI'):
        board_id = 6
    # NeuroMD Devices
    if (eeg_name == 'BrainBit'):
        board_id = 7
    if (eeg_name == 'CallibriEEG'):
        board_id = 9
    if (eeg_name == 'BrainBitBLED'):
        board_id = 18
    # G.TEC Devices
    if (eeg_name == 'Unicorn'):
        board_id = 8
    # Neurosity Devices
    if (eeg_name == 'Notion1'):
        board_id = 13
    if (eeg_name == 'Notion2'):
        board_id = 14
    if (eeg_name == 'Crown'):
        board_id = 23
    # OYMotion
    if (eeg_name == 'GforcePro'):
        board_id = 16
    if (eeg_name == 'GforceDual'):
        board_id = 19
    # FreeEEG32
    if (eeg_name == 'FreeEEG32'):
        board_id = 17
    # Muse
    if (eeg_name == 'MuseSBLED'):
        board_id = 21
    if (eeg_name == 'Muse2BLED'):
        board_id = 22
    if (eeg_name == 'Muse2'):
        board_id = 38
    if (eeg_name == 'MuseS'):
        board_id = 39
    # Ant Neuro
    if (eeg_name == 'AntNeuroBoardEE410'):
        board_id = 24
    if (eeg_name == 'AntNeuroBoardEE411'):
        board_id = 25
    if (eeg_name == 'AntNeuroBoardEE430'):
        board_id = 26
    if (eeg_name == 'AntNeuroBoardEE211'):
        board_id = 27
    if (eeg_name == 'AntNeuroBoardEE212'):
        board_id = 28
    if (eeg_name == 'AntNeuroBoardEE213'):
        board_id = 29
    if (eeg_name == 'AntNeuroBoardEE214'):
        board_id = 30
    if (eeg_name == 'AntNeuroBoardEE215'):
        board_id = 31
    if (eeg_name == 'AntNeuroBoardEE221'):
        board_id = 32
    if (eeg_name == 'AntNeuroBoardEE222'):
        board_id = 33
    if (eeg_name == 'AntNeuroBoardEE223'):
        board_id = 34
    if (eeg_name == 'AntNeuroBoardEE224'):
        board_id = 35
    if (eeg_name == 'AntNeuroBoardEE225'):
        board_id = 36
    # Enophone
    if (eeg_name == 'Enophone'):
        board_id = 37

    # 2. Build the parser object with the correct data based off of the headset
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=False, default=board_id)
    parser.add_argument('--timeout', type=int, help='Seconds timeout for device discovery or connection', required=False,
                        default=TIMEOUT)  # In seconds
    parser.add_argument('--serial-port', type=str,
                        help='serial port', required=False, default=serial_port)
    parser.add_argument('--mac-address', type=str, help='mac address',
                        required=False, default=mac_address)
    parser.add_argument('--ip-address', type=str,
                        help='ip address', required=False, default=ip_address)
    parser.add_argument('--ip-port', type=int,
                        help='ip port', required=False, default=ip_port)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=ip_protocol)
    parser.add_argument('--other-info', type=str,
                        help='other info', required=False, default=other_info)
    parser.add_argument('--serial-number', type=str,
                        help='serial number', required=False, default=serial_number)
    parser.add_argument('--file', type=str, help='file',
                        required=False, default=file)
    parser.add_argument('--streamer-params', type=str,
                        help='streamer params', required=False, default=streamer_params)

    return parser.parse_args()


def getSerialPortBasedOffOfOS():
    serial_port = 'COM3'
    os_name = platform.system()
    if (os_name == 'MacOS'):
        serial_port = '/dev/cu.usbserial-'
    if (os_name == 'Linux'):
        serial_port = "/dev/ttyUSB0"
    return serial_port


def create_eeg_board_object(eeg_params):
    print_debug("Creating EEG Board Object with default data")

    # 1. Enable brainflow loggers
    enable_brainflow_loggers()

    # 2. Determine important EEG params
    brainflow_eeg_params = set_brainflow_input_params(eeg_params)

    # 3. Create the board object with default parameters
    return BoardShim(eeg_params.board_id, brainflow_eeg_params)


def enable_brainflow_loggers():
    BoardShim.enable_board_logger()
    DataFilter.enable_data_logger()
    MLModel.enable_ml_logger()


def set_brainflow_input_params(eeg_params):
    """Gets BrainFlowInputParams object based off of the EEG's custom parameters"""
    bf_params = BrainFlowInputParams()
    bf_params.ip_port = eeg_params.ip_port
    bf_params.serial_port = eeg_params.serial_port
    bf_params.mac_address = eeg_params.mac_address
    bf_params.other_info = eeg_params.other_info
    bf_params.serial_number = eeg_params.serial_number
    bf_params.ip_address = eeg_params.ip_address
    bf_params.ip_protocol = eeg_params.ip_protocol
    bf_params.timeout = eeg_params.timeout
    bf_params.file = eeg_params.file
    return bf_params


def stream_eeg_data(board, eeg_params):
    board_id = board.get_board_id()
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    eeg_channels_count = BoardShim.get_eeg_channels(int(board_id))
    prepare_eeg_stream(board, eeg_params)
    parse_eeg_stream(board, sampling_rate, eeg_channels_count)


def prepare_eeg_stream(board, eeg_params):
    print_debug('Preparing to stream EEG data')
    board.prepare_session()
    board.start_stream(45000, eeg_params.streamer_params)
    BoardShim.log_message(LogLevels.LEVEL_INFO.value,
                          'start sleeping in the main thread')
    # Brainflow recommends waiting at least 4 seconds for the first EEG reading - between this & parse_eeg_stream it waits the 4 secs
    time.sleep(2.7)


def parse_eeg_stream(board, sampling_rate, eeg_channels_count):
    while(True):
        time.sleep(1.3)  # Pause between each EEG Snapshot
        data = board.get_board_data()  # Read what was currently collected on by the board

        # For most of the music generation approaches we just need the "macro" (also known as average) brainwaves' bandpower
        avg_bandpower = get_average_brainwave_bandpower_json(
            data, sampling_rate, eeg_channels_count)

        # For most of the music generation approaches we will need a more specific view
        eeg_channels_of_interest = [0, 1, 2, 3]
        specific_channel_bandpower = get_specific_channels_brainwave_bandpower_json(
            data, sampling_rate, eeg_channels_of_interest)

        merged_output = {**avg_bandpower, **specific_channel_bandpower}
        # writer.writerow(row)

        print(str(json.dumps(merged_output)))
        # Required to flush output for python to allow for python to output script before it has fully executed (which won't happen with the while true)!
        sys.stdout.flush()


def get_average_brainwave_bandpower_json(data, sampling_rate, eeg_channels):
    '''Gets the band values for all of the channels for the EEG'''
    channel_bandpowers = get_channel_brainwave_bandpower(
        data, sampling_rate, eeg_channels)
    avg_brainwave_bandpower = get_average_brainwave_bandpower(
        eeg_channels, channel_bandpowers)
    return {"average_bandpower": avg_brainwave_bandpower}


def get_specific_channels_brainwave_bandpower_json(data, sampling_rate, eeg_channels):
    channel_specific_eeg_data = get_channel_brainwave_bandpower(
        data, sampling_rate, eeg_channels)
    eeg_channels_bandpower = {}
    for i in range(len(eeg_channels)):
        name = 'channel_' + i + 'bandpower'
        eeg_channels_bandpower[name] = channel_specific_eeg_data[i]
    return json.dumps(eeg_channels_bandpower)


def get_channel_brainwave_bandpower(data, sampling_rate, eeg_channels):
    '''Returns the brainwave bandpowers as list for each specific channel in channel range (eeg_channels)'''
    channel_bandpowers = []
    for channel in eeg_channels:
        bandpower = calculate_brainwave_bandpower(data, sampling_rate, channel)
        channel_bandpowers.append(bandpower)
    return channel_bandpowers


def get_average_brainwave_bandpower(eeg_channels, channel_bandpowers):
    '''Returns the average brainwave bandpowers for specified channels (eeg_channels)'''
    avg_brainwave_bandpowers = {}
    for brainwave in BRAINWAVES:
        sum_brainwave_bandpower = 0
        for channel in eeg_channels:
            sum_brainwave_bandpower = sum_brainwave_bandpower + \
                channel_bandpowers[channel][brainwave]
        avg_brainwave_bandpowers[brainwave] = round(
            sum_brainwave_bandpower / len(eeg_channels), ROUNDING_DECIMAL_PLACE)
    return json.dumps(avg_brainwave_bandpowers)


def calculate_brainwave_bandpower(data, sampling_rate, eeg_channel):
    """Gets delta, theta, alpha, beta, & gamma values at current point"""

    df = pd.DataFrame(np.transpose(data))
    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)
    temp = []

    for j in range(0, int(len(df)-1)):
        temp.append(data[eeg_channel][int(j)])

    DataFilter.detrend(np.array(temp), DetrendOperations.LINEAR.value)
    psd = DataFilter.get_psd_welch(data=np.array(temp), nfft=nfft, overlap=nfft //
                                   2, sampling_rate=sampling_rate, window=WindowFunctions.BLACKMAN_HARRIS.value)

    # get_band_power must be in individual try excepts since if the finds 0 or less, it will throw an error
    try:
        delta = round(DataFilter.get_band_power(
            psd, 0.5, 4.0), ROUNDING_DECIMAL_PLACE)
    except:
        delta = 0.0
    try:
        theta = round(DataFilter.get_band_power(
            psd, 4.1, 7.9), ROUNDING_DECIMAL_PLACE)
    except:
        theta = 0.0
    try:
        alpha = round(DataFilter.get_band_power(
            psd, 8.0, 13.9), ROUNDING_DECIMAL_PLACE)
    except:
        alpha = 0.0
    try:
        beta = round(DataFilter.get_band_power(
            psd, 14.0, 31.0), ROUNDING_DECIMAL_PLACE)
    except:
        beta = 0.0
    try:
        gamma = round(DataFilter.get_band_power(
            psd, 32.0, 100.0), ROUNDING_DECIMAL_PLACE)
    except:
        gamma = 0.0

    all_bandpower = {"delta": delta, "theta": theta,
                     "alpha": alpha, "beta": beta, "gamma": gamma}

    print_debug(str(all_bandpower))
    return all_bandpower


def calculate_feature_vector(data, eeg_channels_count, sampling_rate):
    bands = DataFilter.get_avg_band_powers(
        data, eeg_channels_count, sampling_rate, True)
    feature_vector = np.concatenate((bands[0], bands[1]))
    return feature_vector


def print_debug(string):
    debug = 1
    # print(string)
    # sys.stderr.write(string)


if __name__ == "__main__":
    main()
