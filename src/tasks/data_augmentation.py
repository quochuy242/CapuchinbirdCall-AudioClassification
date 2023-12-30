from pydub import effects
from pydub.generators import WhiteNoise
def add_white_noise(sound, decibels = 50):
    """
    Adding White Noise to an Audio Clip and return the new clip
    :param sound:
    :param decibels:
    :return combined:
    """
    noise = WhiteNoise().to_audio_segment(duration=len(sound)) - decibels
    combined = sound.overlay(noise)
    return combined

def normalize_volume(sound):
    """
    Normalize the Volume of a Clip and return the new clip
    Note: Sound should be an AudioSegment
    :param sound:
    :return normalized_sound:
    """
    normalize_sound = effects.normalize(sound)
    return normalize_sound

def filter_out_high_frequency(sound, cut_off = 8e3):
    """
    Filter out High Frequencies in a Clip and return the new clip
    Note: Sound should be an AudioSegment and cut_off is in Hz (default is 8kHz)
    :param sound:
    :param cut_off:
    :return filtered_sound:
    """
    filtered_sound = effects.low_pass_filter(sound, cut_off)
    return filtered_sound

def filter_out_low_frequency(sound, cut_off = 8e3):
    """
    Filter out Low Frequencies in a Clip and return the new clip
    Note: Sound should be an AudioSegment and cut_off is in Hz (default is 8kHz)
    :param sound:
    :param cut_off:
    :return filtered_sound:
    """
    filtered_sound = effects.high_pass_filter(sound, cut_off)
    return filtered_sound