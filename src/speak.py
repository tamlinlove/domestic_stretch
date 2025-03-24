#!/usr/bin/env python3

"""
This module implements the text-to-speech functionality. Using a ROS Action, it implements the speak action server. The action server takes a string and uses the Google Text-to-Speech API to generate an audio file. Then, it uses the audio output device to play the audio file.
"""

import time
import rospy
import resampy
import tempfile
import soundfile as sf
import sounddevice as sd
from gtts import gTTS
from picker_demo.srv import Speak, SpeakRequest, SpeakResponse

NODE_NAME = "speech_manager"
AUDIO_DEVICE = 0

# Services
SPEAK_SERVICE = f"/{NODE_NAME}/speak"


class SpeechManager:

    def __init__(self):
        # ROS
        rospy.init_node(NODE_NAME)

        # Services
        rospy.loginfo("Starting speak action server...")
        rospy.Service(SPEAK_SERVICE, Speak, self.speak_callback)
        rospy.loginfo("Speech node ready.")

    def text_to_speech(self, text: str, language: str = "en"):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as temp_file:
            # Create text to speech and save to file
            tts = gTTS(text=text, lang=language)
            tts.save(temp_file.name)

            start = time.time()

            # Read audio file
            data, sample_rate = sf.read(temp_file.name)
            rospy.logdebug(data.shape, sample_rate)

            # Print time to read audio file
            end = time.time()
            rospy.loginfo(f"Time to read audio file: {end - start}")

            start = time.time()

            # Resample audio if necessary
            if sample_rate != 48000:
                data = resampy.resample(data, sample_rate, 48000)

            # Print time to resample audio
            end = time.time()
            rospy.loginfo(f"Time to resample audio: {end - start}")

            start = time.time()

            # Play audio
            sd.play(data, 48000)
            sd.wait()

            # Print time to play audio
            end = time.time()
            rospy.loginfo(f"Time to play audio: {end - start}")

    def speak_callback(self, msg: SpeakRequest) -> SpeakResponse:
        """
        This function implements the speak action server. It takes a string and uses the Google Text-to-Speech API to generate an audio file. Then, it uses the audio output device to play the audio file.
        """

        # Set audio device
        sd.default.device = AUDIO_DEVICE  # type: ignore
        rospy.loginfo(f"Device: {AUDIO_DEVICE}")

        # Set language
        language = (
            msg.language if msg.language is not None and msg.language != "" else "en"
        )
        rospy.loginfo(f"Language: {language}")

        # Get text
        rospy.loginfo(f"Speaking: {msg.sentence}")
        self.text_to_speech(msg.sentence, language)

        return SpeakResponse(True)


if __name__ == "__main__":
    # Initialize the ROS node.
    rospy.loginfo("Starting speak node...")
    text_to_speech = SpeechManager()
    rospy.spin()
