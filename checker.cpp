
    //Check state vector
    for (unsigned long i = 0; i < state_vec_size; i++) {
        float bit = i%2;
		//complex correct_val;
		complex correct_val = C(1.0f-bit, bit);
        if (state_vec[i] != correct_val) {
           std::cout << "state_vec[" << i << "]: " << state_vec[i] << "\t correct_val: " << correct_val << std::endl;
          throw std::runtime_error("Bad final val in state vec");
       }
  }
