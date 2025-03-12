
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(NonMaxSuppressionTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, selected_indices_size);
  TILING_DATA_FIELD_DEF(uint32_t, batch_size);
  TILING_DATA_FIELD_DEF(uint32_t, num_class);
  TILING_DATA_FIELD_DEF(uint32_t, num_box);
  TILING_DATA_FIELD_DEF(uint32_t, num_box_alin);
  TILING_DATA_FIELD_DEF(uint32_t, center_point_box);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NonMaxSuppression, NonMaxSuppressionTilingData)
}
