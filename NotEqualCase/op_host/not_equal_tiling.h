
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(NotEqualTilingData)
TILING_DATA_FIELD_DEF(uint64_t, smallCoreDataNum);        // 小核处理的总数据量
    TILING_DATA_FIELD_DEF(uint64_t, bigCoreDataNum);          // 大核处理的总数据量
    TILING_DATA_FIELD_DEF(uint64_t, smallCoreCarryNum);       // 小核搬运数据的次数
    TILING_DATA_FIELD_DEF(uint64_t, bigCoreCarryNum);         // 大核搬运数据的次数
    TILING_DATA_FIELD_DEF(uint64_t, tileDataNum);             // 单核能处理最大数据量
    TILING_DATA_FIELD_DEF(uint64_t, smallCoreFinallDealNum);  // 小核最后一次处理的数据量
    TILING_DATA_FIELD_DEF(uint64_t, bigCoreFinallDealNum);    // 大核最后一次处理的数据量
    TILING_DATA_FIELD_DEF(uint64_t, bigCoreNum);              // 大核个数
    TILING_DATA_FIELD_DEF(uint64_t, dataType);                // 运行时数据类型
    TILING_DATA_FIELD_DEF(uint64_t, inputShape00);              // 输入数据维度 后面两位数据第一位表示是第几个输入 第二位表示输入数据第几维
    TILING_DATA_FIELD_DEF(uint64_t, inputShape01);              // 输入数据维度
    TILING_DATA_FIELD_DEF(uint64_t, inputShape10);              // 输入数据维度
    TILING_DATA_FIELD_DEF(uint64_t, inputShape11);              // 输入数据维度
    TILING_DATA_FIELD_DEF(uint64_t, axis);                    // 要广播的维度
    TILING_DATA_FIELD_DEF(uint64_t, who);                     // 谁要进行广播 0 代表第一个输入 1 代表第二个输入
    TILING_DATA_FIELD_DEF(uint64_t, isBroadcast);             // 是否进行广播 0 代表不进行广播 1 代表进行广播
    TILING_DATA_FIELD_DEF(uint64_t, coreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NotEqual, NotEqualTilingData)
}
