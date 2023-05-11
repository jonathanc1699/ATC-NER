atc.far: atc.grm inflight.far taxi.far emergency.far
	thraxcompiler --input_grammar=$< --output_far=$@

inflight.far: inflight.grm example.far exclamation.far numbers.far txt_resource_files/three_digit_num.txt txt_resource_files/two_digit_num.txt txt_resource_files/single_digit_num.txt txt_resource_files/dirs.txt txt_resource_files/alphabets.txt txt_resource_files/radio_freq.txt txt_resource_files/height.txt txt_resource_files/callsigns.txt txt_resource_files/callsigns_2.txt txt_resource_files/heading.txt txt_resource_files/altitude_instructions.txt txt_resource_files/altitude.txt txt_resource_files/speed_instructions.txt txt_resource_files/freq_names.txt txt_resource_files/speed.txt txt_resource_files/SG_waypoints.txt txt_resource_files/time.txt
	thraxcompiler --input_grammar=$< --output_far=$@

example.far: example.grm byte.far numbers.far
	thraxcompiler --input_grammar=$< --output_far=$@

byte.far: byte.grm 
	thraxcompiler --input_grammar=$< --output_far=$@

numbers.far: numbers.grm byte.far
	thraxcompiler --input_grammar=$< --output_far=$@

exclamation.far: exclamation.grm 
	thraxcompiler --input_grammar=$< --output_far=$@

taxi.far: taxi.grm example.far exclamation.far numbers.far txt_resource_files/three_digit_num.txt txt_resource_files/two_digit_num.txt txt_resource_files/single_digit_num.txt txt_resource_files/dirs.txt txt_resource_files/alphabets.txt txt_resource_files/radio_freq.txt txt_resource_files/callsigns.txt txt_resource_files/callsigns_2.txt txt_resource_files/heading.txt txt_resource_files/freq_names.txt txt_resource_files/taxipoint.txt
	thraxcompiler --input_grammar=$< --output_far=$@

emergency.far: emergency.grm example.far exclamation.far numbers.far txt_resource_files/three_digit_num.txt txt_resource_files/two_digit_num.txt txt_resource_files/single_digit_num.txt txt_resource_files/dirs.txt txt_resource_files/alphabets.txt txt_resource_files/altitude_instructions.txt txt_resource_files/altitude.txt txt_resource_files/radio_freq.txt txt_resource_files/callsigns.txt txt_resource_files/SG_waypoints.txt txt_resource_files/callsigns_2.txt txt_resource_files/heading.txt txt_resource_files/freq_names.txt txt_resource_files/emergency_call.txt txt_resource_files/taxipoint.txt
	thraxcompiler --input_grammar=$< --output_far=$@

clean:
	rm -f inflight.far taxi.far emergency.far example.far exclamation.far numbers.far byte.far
