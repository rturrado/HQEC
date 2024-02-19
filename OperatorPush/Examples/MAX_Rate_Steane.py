from OperatorPush.Presets.Heptagon_Steane_Code import setup_heptagon_max_rate_steane
from OperatorPush.ExportToolbox import export_tensor_layer_info_to_csv
from OperatorPush.PushingToolbox import batch_push, batch_push_multiprocessing



if __name__ == '__main__':
    tensor_list = setup_heptagon_max_rate_steane(R=1)

    export_tensor_layer_info_to_csv(tensor_list)

    batch_push_multiprocessing(tensor_list=tensor_list, logger_mode=False)