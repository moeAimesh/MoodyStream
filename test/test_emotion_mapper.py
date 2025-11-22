from detection.emotion_mapper import map_emotion_to_sound


def test_get_sound_for_emotions_happy():
    fake_map = {
        "happy": "sounds/sound_cache/test-happy.mp3"
    }

    key, path = map_emotion_to_sound(["neutral", "happy"], sound_map=fake_map)

    assert key == "happy"
    assert path == "sounds/sound_cache/test-happy.mp3"


def test_get_sound_for_emotions_no_match():
    fake_map = {
        "sad": "sounds/sound_cache/test-sad.mp3"
    }

    key, path = map_emotion_to_sound(["happy"], sound_map=fake_map)

    assert key is None
    assert path is None
