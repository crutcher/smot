### [torch.numel](smot/api_tests/torch_api/torch_shape_ops_test.py)
https://pytorch.org/docs/stable/generated/torch.numel.html

    numel(input) -> int


### [torch.arange](smot/api_tests/torch_api/creation/torch_arange_test.py)
https://pytorch.org/docs/stable/generated/torch.arange.html

    arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor


### [torch.as_strided](smot/api_tests/torch_api/creation/torch_as_strided_test.py)
https://pytorch.org/docs/stable/generated/torch.as_strided.html

    as_strided(input, size, stride, storage_offset=0) -> Tensor


### [torch.as_tensor](smot/api_tests/torch_api/creation/torch_as_tensor_test.py)
https://pytorch.org/docs/stable/generated/torch.as_tensor.html

    as_tensor(data, dtype=None, device=None) -> Tensor


### [torch.complex](smot/api_tests/torch_api/creation/torch_complex_test.py)
https://pytorch.org/docs/stable/generated/torch.complex.html

    complex(real, imag, *, out=None) -> Tensor


### [torch.empty](smot/api_tests/torch_api/creation/torch_empty_test.py)
https://pytorch.org/docs/stable/generated/torch.empty.html

    empty(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False, memory_format=torch.contiguous_format) -> Tensor


### [torch.empty_like](smot/api_tests/torch_api/creation/torch_empty_like_test.py)
https://pytorch.org/docs/stable/generated/torch.empty_like.html

    empty_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor


### [torch.empty_strided](smot/api_tests/torch_api/creation/torch_empty_strided_test.py)
https://pytorch.org/docs/stable/generated/torch.empty_strided.html

    empty_strided(size, stride, *, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False) -> Tensor


### [torch.eye](smot/api_tests/torch_api/creation/torch_eye_test.py)
https://pytorch.org/docs/stable/generated/torch.eye.html

    eye(n, m=None, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor


### [torch.from_numpy](smot/api_tests/torch_api/creation/torch_from_numpy_test.py)
https://pytorch.org/docs/stable/generated/torch.from_numpy.html

    from_numpy(ndarray) -> Tensor


### [torch.frombuffer](smot/api_tests/torch_api/creation/torch_frombuffer_test.py)
https://pytorch.org/docs/stable/generated/torch.frombuffer.html

    frombuffer(buffer, *, dtype, count=-1, offset=0, requires_grad=False) -> Tensor


### [torch.full](smot/api_tests/torch_api/creation/torch_full_test.py)
https://pytorch.org/docs/stable/generated/torch.full.html

    full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor


### [torch.full_like](smot/api_tests/torch_api/creation/torch_full_like_test.py)
https://pytorch.org/docs/stable/generated/torch.full_like.html

    full_like(input, fill_value, \*, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor


### [torch.heaviside](smot/api_tests/torch_api/creation/torch_heaviside_test.py)
https://pytorch.org/docs/stable/generated/torch.heaviside.html

    heaviside(input, values, *, out=None) -> Tensor


### [torch.linspace](smot/api_tests/torch_api/creation/torch_linspace_test.py)
https://pytorch.org/docs/stable/generated/torch.linspace.html

    linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor


### [torch.logspace](smot/api_tests/torch_api/creation/torch_logspace_test.py)
https://pytorch.org/docs/stable/generated/torch.logspace.html

    logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor


### [torch.ones](smot/api_tests/torch_api/creation/torch_ones_test.py)
https://pytorch.org/docs/stable/generated/torch.ones.html

    ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor


### [torch.ones_like](smot/api_tests/torch_api/creation/torch_ones_like_test.py)
https://pytorch.org/docs/stable/generated/torch.ones_like.html

    ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor


### [torch.polar](smot/api_tests/torch_api/creation/torch_polar_test.py)
https://pytorch.org/docs/stable/generated/torch.polar.html

    polar(abs, angle, *, out=None) -> Tensor


### [torch.quantize_per_channel](smot/api_tests/torch_api/creation/torch_quantize_per_channel_test.py)
https://pytorch.org/docs/stable/generated/torch.quantize_per_channel.html

    quantize_per_channel(input, scales, zero_points, axis, dtype) -> Tensor


### [torch.quantize_per_tensor](smot/api_tests/torch_api/creation/torch_quantize_per_tensor_test.py)
https://pytorch.org/docs/stable/generated/torch.quantize_per_tensor.html

    quantize_per_tensor(input, scale, zero_point, dtype) -> Tensor


### [torch.range](smot/api_tests/torch_api/creation/torch_range_test.py)
https://pytorch.org/docs/stable/generated/torch.range.html

    range(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor


### [torch.sparse_coo_tensor](smot/api_tests/torch_api/creation/torch_sparse_coo_tensor_test.py)
https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html

    sparse_coo_tensor(indices, values, size=None, *, dtype=None, device=None, requires_grad=False) -> Tensor


### [torch.tensor](smot/api_tests/torch_api/creation/torch_tensor_test.py)
https://pytorch.org/docs/stable/generated/torch.tensor.html

    tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False) -> Tensor


### [torch.zeros](smot/api_tests/torch_api/creation/torch_zeros_test.py)
https://pytorch.org/docs/stable/generated/torch.zeros.html

    zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor


### [torch.zeros_like](smot/api_tests/torch_api/creation/torch_zeros_like_test.py)
https://pytorch.org/docs/stable/generated/torch.zeros_like.html

    zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor


### [torch.cat](smot/api_tests/torch_api/restructure/torch_cat_test.py)
https://pytorch.org/docs/stable/generated/torch.cat.html

    cat(tensors, dim=0, *, out=None) -> Tensor


### [torch.chunk](smot/api_tests/torch_api/restructure/torch_chunk_test.py)
https://pytorch.org/docs/stable/generated/torch.chunk.html

    chunk(input, chunks, dim=0) -> List of Tensors


### [torch.column_stack](smot/api_tests/torch_api/restructure/torch_column_stack_test.py)
https://pytorch.org/docs/stable/generated/torch.column_stack.html

    column_stack(tensors, *, out=None) -> Tensor


### [torch.concat](smot/api_tests/torch_api/restructure/torch_concat_test.py)
https://pytorch.org/docs/stable/generated/torch.concat.html

    concat(tensors, dim=0, *, out=None) -> Tensor


### [torch.conj](smot/api_tests/torch_api/restructure/torch_conj_test.py)
https://pytorch.org/docs/stable/generated/torch.conj.html

    conj(input) -> Tensor


### [torch.dsplit](smot/api_tests/torch_api/restructure/torch_dsplit_test.py)
https://pytorch.org/docs/stable/generated/torch.dsplit.html

    dsplit(input, indices_or_sections) -> List of Tensors


### [torch.dstack](smot/api_tests/torch_api/restructure/torch_dstack_test.py)
https://pytorch.org/docs/stable/generated/torch.dstack.html

    dstack(tensors, *, out=None) -> Tensor


### [torch.gather](smot/api_tests/torch_api/restructure/torch_gather_test.py)
https://pytorch.org/docs/stable/generated/torch.gather.html

    gather(input, dim, index, *, sparse_grad=False, out=None) -> Tensor


### [torch.hsplit](smot/api_tests/torch_api/restructure/torch_hsplit_test.py)
https://pytorch.org/docs/stable/generated/torch.hsplit.html

    hsplit(input, indices_or_sections) -> List of Tensors


### [torch.index_select](smot/api_tests/torch_api/restructure/torch_index_select_test.py)
https://pytorch.org/docs/stable/generated/torch.index_select.html

    index_select(input, dim, index, *, out=None) -> Tensor


### [torch.masked_select](smot/api_tests/torch_api/restructure/torch_masked_select_test.py)
https://pytorch.org/docs/stable/generated/torch.masked_select.html

    masked_select(input, mask, *, out=None) -> Tensor


### [torch.moveaxis](smot/api_tests/torch_api/restructure/torch_moveaxis_test.py)
https://pytorch.org/docs/stable/generated/torch.moveaxis.html

    moveaxis(input, source, destination) -> Tensor


### [torch.movedim](smot/api_tests/torch_api/restructure/torch_movedim_test.py)
https://pytorch.org/docs/stable/generated/torch.movedim.html

    movedim(input, source, destination) -> Tensor


### [torch.narrow](smot/api_tests/torch_api/restructure/torch_narrow_test.py)
https://pytorch.org/docs/stable/generated/torch.narrow.html

    narrow(input, dim, start, length) -> Tensor


### [torch.nonzero](smot/api_tests/torch_api/restructure/torch_nonzero_test.py)
https://pytorch.org/docs/stable/generated/torch.nonzero.html

    nonzero(input, *, out=None, as_tuple=False) -> LongTensor or tuple of LongTensors


### [torch.permute](smot/api_tests/torch_api/restructure/torch_permute_test.py)
https://pytorch.org/docs/stable/generated/torch.permute.html

    permute(input, dims) -> Tensor


### [torch.reshape](smot/api_tests/torch_api/restructure/torch_reshape_test.py)
https://pytorch.org/docs/stable/generated/torch.reshape.html

    reshape(input, shape) -> Tensor


### [torch.row_stack](smot/api_tests/torch_api/restructure/torch_row_stack_test.py)
https://pytorch.org/docs/stable/generated/torch.row_stack.html

    row_stack(tensors, *, out=None) -> Tensor


### [torch.scatter](smot/api_tests/torch_api/restructure/torch_scatter_test.py)
https://pytorch.org/docs/stable/generated/torch.scatter.html

    scatter(input, dim, index, src) -> Tensor


### [torch.scatter_add](smot/api_tests/torch_api/restructure/torch_scatter_add_test.py)
https://pytorch.org/docs/stable/generated/torch.scatter_add.html

    scatter_add(input, dim, index, src) -> Tensor


### [torch.split](smot/api_tests/torch_api/restructure/torch_split_test.py)
https://pytorch.org/docs/stable/generated/torch.split.html

    Splits the tensor into chunks. Each chunk is a view of the original tensor.


### [torch.squeeze](smot/api_tests/torch_api/restructure/torch_squeeze_test.py)
https://pytorch.org/docs/stable/generated/torch.squeeze.html

    squeeze(input, dim=None, *, out=None) -> Tensor


### [torch.stack](smot/api_tests/torch_api/restructure/torch_stack_test.py)
https://pytorch.org/docs/stable/generated/torch.stack.html

    stack(tensors, dim=0, *, out=None) -> Tensor


### [torch.swapaxes](smot/api_tests/torch_api/restructure/torch_swapaxes_test.py)
https://pytorch.org/docs/stable/torch.html

    swapaxes(input, axis0, axis1) -> Tensor


### [torch.swapdims](smot/api_tests/torch_api/restructure/torch_swapdims_test.py)
https://pytorch.org/docs/stable/torch.html

    swapdims(input, dim0, dim1) -> Tensor


### [torch.t](smot/api_tests/torch_api/restructure/torch_t_test.py)
https://pytorch.org/docs/stable/generated/torch.t.html

    t(input) -> Tensor


### [torch.take](smot/api_tests/torch_api/restructure/torch_take_test.py)
https://pytorch.org/docs/stable/generated/torch.take.html

    take(input, index) -> Tensor


### [torch.take_along_dim](smot/api_tests/torch_api/restructure/torch_take_along_dim_test.py)
https://pytorch.org/docs/stable/generated/torch.take_along_dim.html

    take_along_dim(input, indices, dim, *, out=None) -> Tensor


### [torch.tensor_split](smot/api_tests/torch_api/restructure/torch_tensor_split_test.py)
https://pytorch.org/docs/stable/torch.html

    tensor_split(input, indices_or_sections, dim=0) -> List of Tensors


### [torch.transpose](smot/api_tests/torch_api/restructure/torch_transpose_test.py)
https://pytorch.org/docs/stable/torch.html

    transpose(input, dim0, dim1) -> Tensor


### [torch.vstack](smot/api_tests/torch_api/restructure/torch_vstack_test.py)
https://pytorch.org/docs/stable/generated/torch.vstack.html

    vstack(tensors, *, out=None) -> Tensor


