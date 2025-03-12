
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SoftmaxTilingData)
  TILING_DATA_FIELD_DEF(int32_t, dim);
  TILING_DATA_FIELD_DEF(int32_t, dim_num);
  TILING_DATA_FIELD_DEF(uint32_t, daType);
  TILING_DATA_FIELD_DEF(uint32_t, blockLength);
  TILING_DATA_FIELD_DEF(uint32_t, batch_size);
  TILING_DATA_FIELD_DEF(uint32_t, height);
  TILING_DATA_FIELD_DEF(uint32_t, forelength);
  TILING_DATA_FIELD_DEF(uint32_t, width);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Softmax, SoftmaxTilingData)
}
