"""Optional TenSEAL CKKS helper. Install tenseal to enable encrypted vectors.

The private context/key must remain at the trusted client; never send it to the
aggregation server. This module encrypts flattened numeric updates only.
"""
class TenSEALUnavailable(ImportError): pass


def create_client_context(poly_modulus_degree=8192, coeff_mod_bit_sizes=None):
    try:
        import tenseal as ts
    except ImportError as exc:
        raise TenSEALUnavailable("Install optional dependency: pip install tenseal") from exc
    coeff_mod_bit_sizes = coeff_mod_bit_sizes or [60, 40, 40, 60]
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree, -1, coeff_mod_bit_sizes)
    context.global_scale = 2 ** 40
    context.generate_galois_keys()
    return context


def encrypt_vector(context, values):
    import tenseal as ts
    return ts.ckks_vector(context, [float(v) for v in values])


def serialize_ciphertext(ciphertext):
    return ciphertext.serialize()


def load_ciphertext(public_context, payload):
    import tenseal as ts
    return ts.ckks_vector_from(public_context, payload)


def add_ciphertexts(ciphertexts):
    if not ciphertexts:
        raise ValueError("at least one ciphertext is required")
    result = ciphertexts[0]
    for item in ciphertexts[1:]: result += item
    return result
