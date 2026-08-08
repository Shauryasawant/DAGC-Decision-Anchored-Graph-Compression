import dagc


def test_rescue_features_are_available_from_top_level_package():
    from dagc import RescueEngine, ShadowBuffer, reset_rescue_session

    assert dagc.RescueEngine is RescueEngine
    assert dagc.ShadowBuffer is ShadowBuffer
    assert dagc.reset_rescue_session is reset_rescue_session

    reset_rescue_session("test-session")
    engine = RescueEngine()
    shadow = ShadowBuffer(max_turns=5)

    assert isinstance(engine, RescueEngine)
    assert isinstance(shadow, ShadowBuffer)
