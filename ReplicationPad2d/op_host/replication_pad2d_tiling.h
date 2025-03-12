
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ReplicationPad2dTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, x_total);
  TILING_DATA_FIELD_DEF(uint32_t, y_total);
  TILING_DATA_FIELD_DEF(uint32_t, dim_num);
  TILING_DATA_FIELD_DEF(uint32_t, datatype);
  TILING_DATA_FIELD_DEF(uint32_t, param_c);
  TILING_DATA_FIELD_DEF(uint32_t, param_h);
  TILING_DATA_FIELD_DEF(uint32_t, param_w);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReplicationPad2d, ReplicationPad2dTilingData)
}
