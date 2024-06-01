'''
A compatibility layer for DSS C-API that mimics the official OpenDSS COM interface.

Copyright (c) 2016-2022 Paulo Meira

Copyright (c) 2018-2022 DSS-Extensions contributors
'''
from ._cffi_api_util import Base
from ._types import Float64Array, Int32Array
from typing import AnyStr

class ISettings(Base):
    __slots__ = []

    _columns = [
        'Trapezoidal',
        'LossRegs',
        'VoltageBases',
        'ZoneLock',
        'EmergVminpu',
        'PriceSignal',
        'CktModel',
        'UEregs',
        'UEweight',
        'PriceCurve',
        'NormVminpu',
        'LossWeight',
        'EmergVmaxpu',
        'AutoBusList',
        'NormVmaxpu',
        'AllowDuplicates',
        'ControlTrace',
        'LoadsTerminalCheck',
        'IterateDisabled',
    ]

    @property
    def AllowDuplicates(self) -> bool:
        '''
        {True | False*} Designates whether to allow duplicate names of objects
        
        **NOTE**: for DSS-Extensions, we are considering removing this option in a future 
        release since it has performance impacts even when not used.
        '''
        return self.CheckForError(self._lib.Settings_Get_AllowDuplicates()) != 0

    @AllowDuplicates.setter
    def AllowDuplicates(self, Value: bool):
        self.CheckForError(self._lib.Settings_Set_AllowDuplicates(Value))

    @property
    def AutoBusList(self) -> str:
        '''List of Buses or (File=xxxx) syntax for the AutoAdd solution mode.'''
        return self._get_string(self.CheckForError(self._lib.Settings_Get_AutoBusList()))

    @AutoBusList.setter
    def AutoBusList(self, Value: AnyStr):
        if type(Value) is not bytes:
            Value = Value.encode(self._api_util.codec)

        self.CheckForError(self._lib.Settings_Set_AutoBusList(Value))

    @property
    def CktModel(self) -> int:
        '''{dssMultiphase (0) * | dssPositiveSeq (1) } Indicate if the circuit model is positive sequence.'''
        return self.CheckForError(self._lib.Settings_Get_CktModel())

    @CktModel.setter
    def CktModel(self, Value: int):
        self.CheckForError(self._lib.Settings_Set_CktModel(Value))

    @property
    def ControlTrace(self) -> bool:
        '''{True | False*} Denotes whether to trace the control actions to a file.'''
        return self.CheckForError(self._lib.Settings_Get_ControlTrace()) != 0

    @ControlTrace.setter
    def ControlTrace(self, Value: bool):
        self.CheckForError(self._lib.Settings_Set_ControlTrace(Value))

    @property
    def EmergVmaxpu(self) -> float:
        '''Per Unit maximum voltage for Emergency conditions.'''
        return self.CheckForError(self._lib.Settings_Get_EmergVmaxpu())

    @EmergVmaxpu.setter
    def EmergVmaxpu(self, Value: float):
        self.CheckForError(self._lib.Settings_Set_EmergVmaxpu(Value))

    @property
    def EmergVminpu(self) -> float:
        '''Per Unit minimum voltage for Emergency conditions.'''
        return self.CheckForError(self._lib.Settings_Get_EmergVminpu())

    @EmergVminpu.setter
    def EmergVminpu(self, Value: float):
        self.CheckForError(self._lib.Settings_Set_EmergVminpu(Value))

    @property
    def LossRegs(self) -> Int32Array:
        '''Integer array defining which energy meter registers to use for computing losses'''
        self.CheckForError(self._lib.Settings_Get_LossRegs_GR())
        return self._get_int32_gr_array()

    @LossRegs.setter
    def LossRegs(self, Value: Int32Array):
        Value, ValuePtr, ValueCount = self._prepare_int32_array(Value)
        self.CheckForError(self._lib.Settings_Set_LossRegs(ValuePtr, ValueCount))

    @property
    def LossWeight(self) -> float:
        '''Weighting factor applied to Loss register values.'''
        return self.CheckForError(self._lib.Settings_Get_LossWeight())

    @LossWeight.setter
    def LossWeight(self, Value: float):
        self.CheckForError(self._lib.Settings_Set_LossWeight(Value))

    @property
    def NormVmaxpu(self) -> float:
        '''Per Unit maximum voltage for Normal conditions.'''
        return self.CheckForError(self._lib.Settings_Get_NormVmaxpu())

    @NormVmaxpu.setter
    def NormVmaxpu(self, Value: float):
        self.CheckForError(self._lib.Settings_Set_NormVmaxpu(Value))

    @property
    def NormVminpu(self) -> float:
        '''Per Unit minimum voltage for Normal conditions.'''
        return self.CheckForError(self._lib.Settings_Get_NormVminpu())

    @NormVminpu.setter
    def NormVminpu(self, Value: float):
        self.CheckForError(self._lib.Settings_Set_NormVminpu(Value))

    @property
    def PriceCurve(self) -> str:
        '''Name of LoadShape object that serves as the source of price signal data for yearly simulations, etc.'''
        return self._get_string(self.CheckForError(self._lib.Settings_Get_PriceCurve()))

    @PriceCurve.setter
    def PriceCurve(self, Value: AnyStr):
        if type(Value) is not bytes:
            Value = Value.encode(self._api_util.codec)

        self.CheckForError(self._lib.Settings_Set_PriceCurve(Value))

    @property
    def PriceSignal(self) -> float:
        '''Price Signal for the Circuit'''
        return self.CheckForError(self._lib.Settings_Get_PriceSignal())

    @PriceSignal.setter
    def PriceSignal(self, Value: float):
        self.CheckForError(self._lib.Settings_Set_PriceSignal(Value))

    @property
    def Trapezoidal(self) -> bool:
        '''Gets value of trapezoidal integration flag in energy meters. Defaults to `False`.'''
        return self.CheckForError(self._lib.Settings_Get_Trapezoidal()) != 0

    @Trapezoidal.setter
    def Trapezoidal(self, Value: bool):
        self.CheckForError(self._lib.Settings_Set_Trapezoidal(Value))

    @property
    def UEregs(self) -> Int32Array:
        '''Array of Integers defining energy meter registers to use for computing UE'''
        self.CheckForError(self._lib.Settings_Get_UEregs_GR())
        return self._get_int32_gr_array()

    @UEregs.setter
    def UEregs(self, Value: Int32Array):
        Value, ValuePtr, ValueCount = self._prepare_int32_array(Value)
        self.CheckForError(self._lib.Settings_Set_UEregs(ValuePtr, ValueCount))

    @property
    def UEweight(self) -> float:
        '''Weighting factor applied to UE register values.'''
        return self.CheckForError(self._lib.Settings_Get_UEweight())

    @UEweight.setter
    def UEweight(self, Value: float):
        self.CheckForError(self._lib.Settings_Set_UEweight(Value))

    @property
    def VoltageBases(self) -> Float64Array:
        '''Array of doubles defining the legal voltage bases in kV L-L'''
        self.CheckForError(self._lib.Settings_Get_VoltageBases_GR())
        return self._get_float64_gr_array()

    @VoltageBases.setter
    def VoltageBases(self, Value: Float64Array):
        Value, ValuePtr, ValueCount = self._prepare_float64_array(Value)
        self.CheckForError(self._lib.Settings_Set_VoltageBases(ValuePtr, ValueCount))

    @property
    def ZoneLock(self) -> bool:
        '''{True | False*}  Locks Zones on energy meters to prevent rebuilding if a circuit change occurs.'''
        return self.CheckForError(self._lib.Settings_Get_ZoneLock()) != 0

    @ZoneLock.setter
    def ZoneLock(self, Value: bool):
        self.CheckForError(self._lib.Settings_Set_ZoneLock(Value))

    @property
    def AllocationFactors(self):
        '''(write-only) Sets all load allocation factors for all loads defined by XFKVA property to this value.'''
        raise AttributeError("This property is write-only!")

    @AllocationFactors.setter
    def AllocationFactors(self, Value: float):
        self.CheckForError(self._lib.Settings_Set_AllocationFactors(Value))

    @property
    def LoadsTerminalCheck(self) -> bool:
        '''
        Controls whether the terminals are checked when updating the currents in Load component. Defaults to True.
        If the loads are guaranteed to have their terminals closed throughout the simulation, this can be set to False to save some time.
        
        (API Extension)
        '''
        return self.CheckForError(self._lib.Settings_Get_LoadsTerminalCheck()) != 0

    @LoadsTerminalCheck.setter
    def LoadsTerminalCheck(self, Value: bool):
        self.CheckForError(self._lib.Settings_Set_LoadsTerminalCheck(Value))
        
    @property
    def IterateDisabled(self) -> int:
        '''
        Controls whether `First`/`Next` iteration includes or skips disabled circuit elements.
        The default behavior from OpenDSS is to skip those. The user can still activate the element by name or index.
        
        The default value for IterateDisabled is 0, keeping the original behavior.
        Set it to 1 (or `True`) to include disabled elements.
        Other numeric values are reserved for other potential behaviors.
        
        (API Extension)
        '''
        return self.CheckForError(self._lib.Settings_Get_IterateDisabled())

    @IterateDisabled.setter
    def IterateDisabled(self, Value: int):
        self.CheckForError(self._lib.Settings_Set_IterateDisabled(Value))
