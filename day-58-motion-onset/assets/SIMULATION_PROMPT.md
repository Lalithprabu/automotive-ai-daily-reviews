# Simulation script / prompt (image-video AI or rendering engine)
16:9 (1240x640), ~10 s at 6 fps, light technical-dashboard look, title "AI in action: will that parked/static road user start moving?".
**Left (top-down kerbside, INPUT -> PREDICTION -> GROUND TRUTH):** grey road band along the bottom with a blue EGO car driving left to right. Above it, 14 road users as coloured boxes sized by class (parked car, delivery van, pedestrian, cyclist, e-scooter, cone). Static ones are semi-transparent; a red halo around each static one grows and thickens with the predicted probability that it starts moving within 3 s, with a small label "class p". A dashed GREEN ring around a static object = it really starts moving within 3 s (GROUND TRUTH). Once an object moves it turns opaque, gets a black velocity arrow and a short green trail of its recent path.
**Right top:** horizontal bars, top-7 onset probabilities, green bar = truly moves in 3 s, grey otherwise.
**Right middle:** time series P(onset) for the 'star' object: blue = recurrent GNN, orange = stateless GNN, green shaded band = frames where onset is within 3 s, dashed green line = true onset.
**Right bottom:** ego speed advisory (m/s) = 8*(1 - 0.75*max onset risk ahead), line dipping when a risky object is within 14 m ahead.
Footer: "Synthetic demo - NOT the paper's data nor real driving data".
