#include "seal/seal.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace seal;

void save_to_csv(const string &filename, vector<pair<string, double>> &data) {
    ofstream file(filename);
    file << "Operation,Noise Budget\n";
    for (auto &entry : data) {
        file << entry.first << "," << entry.second << "\n";
    }
    file.close();
}

int main() {
    EncryptionParameters parms(scheme_type::bfv);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
    parms.set_plain_modulus(256);

    SEALContext context(parms);

    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);
    SecretKey secret_key = keygen.secret_key();
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);

    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);
    BatchEncoder encoder(context);

    Plaintext plain1, plain2;
    vector<uint64_t> vec1(encoder.slot_count(), 5ULL);
    vector<uint64_t> vec2(encoder.slot_count(), 3ULL);
    encoder.encode(vec1, plain1);
    encoder.encode(vec2, plain2);

    Ciphertext encrypted1, encrypted2;
    encryptor.encrypt(plain1, encrypted1);
    encryptor.encrypt(plain2, encrypted2);

    vector<pair<string, double>> noise_data;
    noise_data.emplace_back("Initial", decryptor.invariant_noise_budget(encrypted1));

    evaluator.add_inplace(encrypted1, encrypted2);
    noise_data.emplace_back("After Add", decryptor.invariant_noise_budget(encrypted1));

    evaluator.multiply_inplace(encrypted1, encrypted2);
    evaluator.relinearize_inplace(encrypted1, relin_keys);
    noise_data.emplace_back("After Multiply", decryptor.invariant_noise_budget(encrypted1));

    save_to_csv("bgv_noise_data.csv", noise_data);
    cout << "Noise data saved to bgv_noise_data.csv" << endl;
    return 0;
}
