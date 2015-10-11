#!/bin/bash

(
python './semi_automated_extraction_features_chicago_data.py' 2 0
)&

(
python './semi_automated_extraction_features_chicago_data.py' 2 1
)&

wait


