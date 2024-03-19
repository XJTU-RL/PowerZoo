unit CktElement;

// ----------------------------------------------------------
// Copyright (c) 2008-2021, Electric Power Research Institute, Inc.
// All rights reserved.
// ----------------------------------------------------------

interface

uses
    Classes,
    UComplex, DSSUcomplex,
    Ucmatrix,
    ArrayDef,
    Terminal,
    DSSObject,
    DSSClass,
    DSSPointerList,
    DSSClassDefs;

type

    TDSSCktElement = class(TDSSObject)
    PROTECTED
        FEnabled: Boolean;
    PRIVATE
        FBusNames: pStringArray; // Bus + Nodes (a.1.2.3.0)
        FActiveTerminal: Int8;
        FYPrimInvalid: Boolean;

        procedure Set_Freq(Value: Double);  // set freq and recompute YPrim.

        procedure Set_Nconds(Value: Int8);
        function Get_ActiveTerminal(): Int8; inline;
        procedure Set_ActiveTerminal(value: Int8);
        function Get_ConductorClosed(Index: Integer): Boolean; inline;
        procedure Set_YprimInvalid(const Value: Boolean);
        function Get_Losses: Complex;   // Get total losses for property...
        function Get_Power(idxTerm: Integer): Complex;    // Get total complex power in active terminal
        function Get_MaxPower(idxTerm: Integer): Complex;    // Get equivalent total complex power in active terminal based on phase with max current
        function Get_MaxCurrent(idxTerm: Integer): Double; // Get equivalent total complex current on phase with max current
        function Get_MaxCurrentAng(idxTerm:Integer): Double; // Get equivalent angle of the total complex current on phase with max current
        function Get_MaxVoltageC(idxTerm: Integer): Complex; // Get equivalent total complex voltage on phase
        function Get_MaxVoltage(idxTerm: Integer): Double; // Get equivalent total complex voltage on phase
        function Get_MaxVoltageAng(idxTerm:Integer): Double; // Get equivalent angle of the total complex voltage on phase
        function Get_PCE_Value(idxTerm:Integer; ValType:Integer): Double; // Get a value for the active PCE such as P, Q, Vmag, IMag, etc.

        procedure DoYprimCalcs(Ymatrix: TCMatrix);

    PUBLIC
        Fnterms: Int8;
        Fnconds: Int8;  // no. conductors per terminal
        Fnphases: Integer;  // Phases, this device -- TODO: Int8 someday...

        ComplexBuffer: pComplexArray;

        IterminalSolutionCount: Integer;

        BusIndex: Integer;
        YPrim_Series,
        YPrim_Shunt,
        YPrim: TCMatrix;   // Order will be NTerms * Ncond
        FYprimFreq: Double;     // Frequency at which YPrim has been computed

        procedure Set_Enabled(Value: Boolean); VIRTUAL;
        procedure Set_ConductorClosed(Index: Integer; Value: Boolean); VIRTUAL;
        procedure Set_NTerms(Value: Int8);
    PUBLIC
        Handle: Integer;

        // Total Noderef array for element
        NodeRef: pIntegerArray;  // Need fast access to this
        Yorder: Integer;

        // LastTerminalChecked: Int8;  // Flag used in tree searches -- UNUSED

        ControlElementList: TDSSPointerList; //Pointer to control for this device

        Iterminal: pComplexArray;  // Others need this
        Vterminal: pComplexArray;

        BaseFrequency: Double;

        Terminals: Array of TPowerTerminal;
        TerminalsChecked: Array of Boolean;
        ActiveTerminal: ^TPowerTerminal;

        PublicDataSize: Integer;  // size of PublicDataStruct
        PublicDataStruct: Pointer;  // Generic Pointer to public data Block that may be access by other classes of elements
                             // Accessing app has to know the structure
                             // Inited to Nil.  If Nil, accessing app should ignore

        constructor Create(ParClass: TDSSClass);
        destructor Destroy; OVERRIDE;
        procedure MakeLike(OtherObj: Pointer); override;
        function FirstBus(): String;
        function NextBus(): String;

        function AllConductorsClosed(): Boolean;
        function GetYPrim(var Ymatrix: TCmatrix; Opt: Integer): Integer; VIRTUAL;  //returns values of array
        function GetYPrimValues(Opt: Integer): pComplexArray; VIRTUAL;
        function MaxTerminalOneIMag: Double;   // Max of Iterminal 1 phase currents
        procedure ComputeIterminal; VIRTUAL;   // Computes Iterminal for this device
        procedure ComputeVterminal;
        procedure ZeroITerminal; inline;
        procedure GetCurrents(Curr: pComplexArray); VIRTUAL; OVERLOAD; ABSTRACT; //Get present value of terminal Curr for reports
        procedure GetCurrents(Curr: ArrayOfComplex); VIRTUAL; OVERLOAD; //Get present value of terminal Curr for reports
        function InjCurrents: Integer; VIRTUAL; // Applies to PC Elements Puts straight into Solution Array

        function GetBus(i: Integer): String;  // Get bus name by index
        procedure SetBus(i: Integer; const s: String); virtual;  // Set bus name by index
        procedure SetNodeRef(iTerm: Integer; NodeRefArray: pIntegerArray); VIRTUAL;  // Set NodeRef Array for fast solution with intrinsics
        procedure RecalcElementData; VIRTUAL; ABSTRACT;
        procedure CalcYPrim; VIRTUAL;

        procedure MakePosSequence(); VIRTUAL;  // Make a positive Sequence Model

        procedure GetTermVoltages(iTerm: Integer; VBuffer: PComplexArray);
        procedure GetPhasePower(PowerBuffer: pComplexArray); VIRTUAL;
        procedure GetPhaseLosses(var Num_Phases: Integer; LossBuffer: pComplexArray); VIRTUAL;
        procedure GetLosses(var TotalLosses, LoadLosses, NoLoadLosses: Complex); VIRTUAL;
        procedure GetSeqLosses(var PosSeqLosses, NegSeqLosses, ZeroModeLosses: complex); VIRTUAL;

        procedure DumpProperties(F: TStream; Complete: Boolean; Leaf: Boolean = False); OVERRIDE;

        property Enabled: Boolean READ FEnabled WRITE Set_Enabled;
        property YPrimInvalid: Boolean READ FYPrimInvalid WRITE set_YprimInvalid;
        property YPrimFreq: Double READ FYprimFreq WRITE Set_Freq;
        property NTerms: Int8 READ Fnterms WRITE Set_NTerms;
        property NConds: Int8 READ Fnconds WRITE Set_Nconds;
        property NPhases: Integer READ Fnphases;
        property Losses: Complex READ Get_Losses;
        property Power[idxTerm: Integer]: Complex READ Get_Power;  // Total power in active terminal
        property MaxPower[idxTerm: Integer]: Complex READ Get_MaxPower;  // Total power in active terminal
        property MaxCurrent[idxTerm: Integer]: Double READ Get_MaxCurrent;  // Max current in active terminal
        property MaxVoltage[idxTerm: Integer]: Double READ Get_MaxVoltage;  // Max voltage in active terminal
        property ActiveTerminalIdx: Int8 READ Get_ActiveTerminal WRITE Set_ActiveTerminal;
        property Closed[Index: Integer]: Boolean READ Get_ConductorClosed WRITE Set_ConductorClosed;
        property MaxCurrentAng[idxTerm: Integer]: Double READ Get_MaxCurrentAng;  // Max current in active terminal
        property MaxVoltageAng[idxTerm: Integer]: Double READ Get_MaxVoltageAng;  // Max current in active terminal
        property PCEValue[Index:Integer; ValType: Integer]: Double READ Get_PCE_Value;

        procedure SumCurrents;
        
        procedure Get_Current_Mags(cMBuffer: pDoubleArray); // Returns the Currents vector in magnitude
    end;


implementation

uses
    DSSGlobals,
    SysUtils,
    Utilities,
    Math,
    Solution,
    DSSHelper,
    DSSObjectHelper,
    TypInfo,
    CktElementClass;

const
    cEpsilon : Complex = (re: EPSILON; im: 0.0);

constructor TDSSCktElement.Create(ParClass: TDSSClass);
begin
    inherited Create(ParClass);

    NodeRef := NIL;
    YPrim_Series := NIL;
    YPrim_Shunt := NIL;
    YPrim := NIL;
    FBusNames := NIL;
    Vterminal := NIL;
    Iterminal := NIL;  // present value of terminal current
    Terminals := NIL;
    TerminalsChecked := NIL;

    ComplexBuffer := NIL;
    PublicDataStruct := NIL;   // pointer to fixed struct of data to be shared
    PublicDataSize := 0;

    Handle := -1;
    BusIndex := 0;
    FNterms := 0;
    Fnconds := 0;
    Fnphases := 0;
    DSSObjType := 0;
    Yorder := 0;

    YPrimInvalid := TRUE;
    FEnabled := TRUE;

    // Make list for a small number of controls with an increment of 1
    ControlElementList := TDSSPointerList.Create(1);

    FActiveTerminal := 0;
    // LastTerminalChecked := 0;

    // Indicates which solution Itemp is computed for
    IterminalSolutionCount := -1;

    BaseFrequency := ActiveCircuit.Fundamental;
end;

destructor TDSSCktElement.Destroy;
var
    i: Integer;
begin
    if DSS = NIL then
    begin
        inherited Destroy;
        exit;
    end;
    for i := 1 to FNTerms do
        FBusNames[i] := ''; // Free up strings

    SetLength(Terminals, 0);
    SetLength(TerminalsChecked, 0);
    Reallocmem(FBusNames, 0);
    Reallocmem(Iterminal, 0);
    Reallocmem(Vterminal, 0);
    Reallocmem(NodeRef, 0);
    Reallocmem(ComplexBuffer, 0);

    if assigned(ControlElementList) then
        ControlElementList.Free;

    // Dispose YPrims
    if (Yprim <> NIL) AND (Yprim <> Yprim_Shunt) AND (Yprim <> Yprim_Series) then
        Yprim.Free;
    if Yprim_Series <> NIL then
        Yprim_Series.Free;
    if Yprim_Shunt <> NIL then
        Yprim_Shunt.Free;

    inherited Destroy;
end;

procedure TDSSCktElement.Set_YprimInvalid(const Value: Boolean);
begin
    FYPrimInvalid := value;
    if Value and FEnabled then
        // If this device is in the circuit, then we have to rebuild Y on a change in Yprim
        ActiveCircuit.Solution.SystemYChanged := TRUE;
end;

function TDSSCktElement.Get_ActiveTerminal(): Int8; inline;
begin
    Result := FActiveTerminal + 1;
end;

procedure TDSSCktElement.Set_ActiveTerminal(value: Int8);
begin
    if (Value > 0) and (Value <= fNterms) then
    begin
        FActiveTerminal := Value - 1;
        ActiveTerminal := @Terminals[FActiveTerminal];
    end;
end;

function TDSSCktElement.Get_ConductorClosed(Index: Integer): Boolean; inline;
// return state of selected conductor
// if index=0 return true if all phases closed, else false
var
    i: Integer;
begin
    if (Index = 0) then
    begin
        Result := TRUE;
        for i := 1 to Fnphases do
        begin
            if not Terminals[FActiveTerminal].ConductorsClosed[i - 1] then
            begin
                Result := FALSE;
                Break;
            end;
        end;
    end
    else
    if (Index > 0) and (Index <= Fnconds) then
        Result := Terminals[FActiveTerminal].ConductorsClosed[Index - 1]
    else
        Result := FALSE;
end;

procedure TDSSCktElement.Set_ConductorClosed(Index: Integer; Value: Boolean);
var
    i: Integer;
begin
    if (Index = 0) then
    begin  // Do all conductors
        for i := 0 to Fnphases - 1 do
            Terminals[FActiveTerminal].ConductorsClosed[i] := Value;
        YPrimInvalid := TRUE; // this also sets the global SystemYChanged flag
    end
    else
    begin
        if (Index > 0) and (Index <= Fnconds) then
        begin
            Terminals[FActiveTerminal].ConductorsClosed[index - 1] := Value;
            YPrimInvalid := TRUE;
        end;
    end;
end;

procedure TDSSCktElement.Set_NConds(Value: Int8);
begin
    // Check for an almost certain programming error
    if Value <= 0 then
    begin
        DoSimpleMsg('Invalid number of terminals (%d) for "%s"',
            [Value, FullName], 749);
        Exit;
    end;

    if Value <> Fnconds then
        ActiveCircuit.BusNameRedefined := TRUE;
    Fnconds := Value;
    Set_Nterms(fNterms);  // ReallocTerminals    NEED MORE EFFICIENT WAY TO DO THIS
end;

procedure TDSSCktElement.Set_NTerms(Value: Int8);
var
    i: Integer;
    NewBusNames: pStringArray;
begin
    // Check for an almost certain programming error
    if Value <= 0 then
    begin
        DoSimpleMsg('Invalid number of terminals (%d) for "%s"',
            [Value, FullName], 749);
        Exit;
    end;

    // If value is same as present value, no reallocation necessary;
    // If either Nterms or Nconds has changed then reallocate
    if (value = FNterms) and ((Value * Fnconds) = Yorder) then
        Exit;
    
    // Sanity Check
    if Fnconds > 101 then
    begin
        DoSimpleMsg('Warning: Number of conductors is very large (%d) for Circuit Element: "%s". Possible error in specifying the Number of Phases for element.',
            [Fnconds, FullName], 750);
    end;


     // ReAllocate BusNames
     // because they are Strings, we have to do it differently

    if Value < fNterms then
        ReallocMem(FBusNames, Sizeof(FBusNames[1]) * Value)  // Keeps old values; truncates storage
    else
    begin
        if FBusNames = NIL then
        begin
            // First allocation
            //  Always allocate  arrays of strings with AllocMem so that the pointers are all nil
            // else Delphi thinks non-zero values are pointing to an existing string.
            FBusNames := AllocMem(Sizeof(FBusNames[1]) * Value); //    fill with zeros or strings will crash
            for i := 1 to Value do
                FBusNames[i] := Name + '_' + IntToStr(i);  // Make up a bus name to stick in.
                 // This is so devices like transformers which may be defined on multiple commands
                 // will have something in the BusNames array.
        end
        else
        begin
            NewBusNames := AllocMem(Sizeof(FBusNames[1]) * Value);  // make some new space
            for i := 1 to fNterms do
                NewBusNames[i] := FBusNames[i];   // copy old into new
            for i := 1 to fNterms do
                FBusNames[i] := '';   // decrement usage counts by setting to nil string
            for i := fNterms + 1 to Value do
                NewBusNames[i] := Name + '_' + IntToStr(i);  // Make up a bus name to stick in.
            ReAllocMem(FBusNames, 0);  // dispose of old array storage
            FBusNames := NewBusNames;
        end;
    end;

    // Reallocate Terminals if Nconds or NTerms changed
    SetLength(Terminals, Value);
    SetLength(TerminalsChecked, Value);
    for i := 1 to Value do
        TerminalsChecked[i - 1] := False;

    FNterms := Value;    // Set new number of terminals
    Yorder := FNterms * Fnconds;
    ReallocMem(Vterminal, Sizeof(Vterminal[1]) * Yorder);
    ReallocMem(Iterminal, Sizeof(Iterminal[1]) * Yorder);
    ReallocMem(ComplexBuffer, Sizeof(ComplexBuffer[1]) * Yorder);    // used by both PD and PC elements

    for i := 1 to Value do
        Terminals[i - 1].Init(Fnconds);
end;

procedure TDSSCktElement.Set_Enabled(Value: Boolean);
//  If disabled, but defined, just have to processBusDefs.  Adding a bus OK
// If being removed from circuit, could remove a node or bus so have to rebuild
begin
    if Value = FEnabled then
        Exit;
        
    FEnabled := Value;
    ActiveCircuit.BusNameRedefined := TRUE;  // forces rebuilding of Y matrix and bus lists
end;

function TDSSCktElement.GetYPrim(var Ymatrix: TCmatrix; Opt: Integer): Integer;
//returns pointer to actual YPrim
begin
    case Opt of
        ALL_YPRIM:
            Ymatrix := Yprim;
        SERIES:
            YMatrix := YPrim_Series;
        SHUNT:
            YMatrix := YPrim_Shunt;
    end;
    Result := 0;
end;

function TDSSCktElement.GetYPrimValues(Opt: Integer): pComplexArray;
// Return a pointer to the Beginning the storage arrays for fast access
var
    Norder: Integer;
begin
    Result := NIL;
    case Opt of
        ALL_YPRIM:
            if YPrim <> NIL then
                Result := Yprim.GetValuesArrayPtr(Norder);
        SERIES:
            if YPrim_Series <> NIL then
                Result := Yprim_Series.GetValuesArrayPtr(Norder);
        SHUNT:
            if YPrim_Shunt <> NIL then
                Result := YPrim_Shunt.GetValuesArrayPtr(Norder);
    end;
end;

procedure TDSSCktElement.GetLosses(var TotalLosses, LoadLosses,
    NoLoadLosses: Complex);
begin
    // For no override, Default behavior is:
    // Just return total losses and set LoadLosses=total losses and noload losses =0

    TotalLosses := Losses;  // Watts, vars
    LoadLosses := TotalLosses;
    NoLoadLosses := 0;
end;

function TDSSCktElement.InjCurrents: Integer;  // Applies to PC Elements
begin
    Result := 0;
    DoErrorMsg(Format(_('Improper call to InjCurrents for Element: "%s".'), [FullName]), '****',
        'Called CktElement class base function instead of actual.', 753)
end;

procedure TDSSCktElement.SetNodeRef(iTerm: Integer; NodeRefArray: pIntegerArray);
// Also allocates VTemp  & Itemp
var
    Size, Size2: Integer;
begin
    // Allocate NodeRef and move new values into it.
    Size := Yorder * SizeOf(NodeRef[1]);
    Size2 := SizeOf(NodeRef[1]) * Fnconds;  // Size for one terminal
    ReallocMem(NodeRef, Size);  // doesn't do anything if already properly allocated
    Move(NodeRefArray[1], NodeRef[(iTerm - 1) * Fnconds + 1], Size2);  // Zap
    Move(NodeRefArray[1], Terminals[iTerm - 1].TermNodeRef[0], Size2);  // Copy in Terminal as well

    // Allocate temp array used to hold voltages and currents for calcs
    ReallocMem(Vterminal, Yorder * SizeOf(Vterminal[1]));
    ReallocMem(Iterminal, Yorder * SizeOf(Iterminal[1]));
    ReallocMem(ComplexBuffer, Yorder * SizeOf(ComplexBuffer[1]));
end;

function TDSSCktElement.FirstBus(): String;
begin
    if FNterms > 0 then
    begin
        BusIndex := 1;
        Result := FBusNames[BusIndex];
    end
    else
        Result := '';
end;

function TDSSCktElement.NextBus(): String;
begin
    Result := '';
    if FNterms > 0 then
    begin
        Inc(BusIndex);
        if BusIndex <= FNterms then
            Result := FBusNames[BusIndex]
        else
            BusIndex := FNterms;
    end;
end;

function TDSSCktElement.GetBus(i: Integer): String;  // Get bus name by index

begin
    if i <= FNTerms then
        Result := FBusNames[i]
    else
        Result := '';
end;

procedure TDSSCktElement.SetBus(i: Integer; const s: String); // Set bus name by index
begin
    if i <= FNterms then
    begin
        FBusNames[i] := AnsiLowerCase(S);
        ActiveCircuit.BusNameRedefined := TRUE;  // Set Global Flag to signal circuit to rebuild busdefs
    end
    else
        DoSimpleMsg('Attempt to set bus name for non-existent circuit element terminal (%d): "%s"', [i, s], 7541);
end;

procedure TDSSCktElement.Set_Freq(Value: Double);
begin
    if Value > 0.0 then
        FYprimFreq := Value;
end;

procedure TDSSCktElement.CalcYPrim;
begin
    if YPrim_Series <> NIL then
        DoYPrimCalcs(Yprim_Series);
    if YPrim_Shunt <> NIL then
        DoYPrimCalcs(YPrim_Shunt);
    if YPrim <> NIL then
        DoYPrimCalcs(YPrim);

{$IFDEF DSS_CAPI_INCREMENTAL_Y}
    if ((ActiveCircuit.Solution.SolverOptions and ord(TSolverOptions.AlwaysResetYPrimInvalid)) <> 0) then
        YPrimInvalid := False;
{$ENDIF}
end;

procedure TDSSCktElement.ComputeIterminal;
begin
    // to save time, only recompute if a different solution than last time it was computed.
    if IterminalSolutionCount <> ActiveCircuit.Solution.SolutionCount then
    begin
        GetCurrents(Iterminal);
        IterminalSolutionCount := ActiveCircuit.Solution.SolutionCount;
    end;
end;

function TDSSCktElement.MaxTerminalOneIMag: Double;
// Get max of phase currents on the first terminal; Requires computing Iterminal
var
    i: Integer;
begin
    Result := 0.0;
    if (not FEnabled) or (NodeRef = NIL) then
        Exit;
        
    for i := 1 to Fnphases do
        Result := Max(Result, SQR(Iterminal[i].re) + SQR(Iterminal[i].im));
            
    Result := Sqrt(Result);  // just do the sqrt once and save a little time
end;

procedure TDSSCktElement.Get_Current_Mags(cMBuffer: pDoubleArray);
var
    i: Integer;
begin
    for i := 1 to Fnphases do
        cMBuffer[i] := cabs(Iterminal[i]);
end;

function TDSSCktElement.Get_Power(idxTerm: Integer): Complex;    // Get total complex power in active terminal
var
    i, k, n: Integer;
    NodeV: pNodeVarray;
begin
    Result := 0;
    ActiveTerminalIdx := idxTerm;
    if (not FEnabled) or (NodeRef = NIL) then
        Exit;
        
    ComputeIterminal();

    // Method: Sum complex power going into phase conductors of active terminal
    NodeV := ActiveCircuit.Solution.NodeV;
    k := (idxTerm - 1) * Fnconds;
    for i := 1 to Fnconds do     // 11-7-08 Changed from Fnphases - was not accounting for all conductors
    begin
        n := ActiveTerminal^.TermNodeRef[i - 1]; // don't bother for grounded node
        if n > 0 then
            Result += NodeV[n] * cong(Iterminal[k + i]);
    end;
    // If this is a positive sequence circuit, then we need to multiply by 3 to get the 3-phase power
    if ActiveCircuit.PositiveSequence then
        Result := Result * 3.0;
end;

function TDSSCktElement.Get_Losses: Complex;
// get total losses in circuit element, all phases, all terminals.
// Returns complex losses (watts, vars)
var
    i, j, k, n: Integer;
    NodeV: pNodeVarray;
begin
    Result := 0;
    if (not FEnabled) or (NodeRef = NIL) then
        Exit;
        
    ComputeIterminal();

    // Method: Sum complex power going into all conductors of all terminals
    // Special for AutoTransformer - sum based on NPhases rather then Yorder

    NodeV := ActiveCircuit.Solution.NodeV;
    if (CLASSMASK and self.DSSObjType) = AUTOTRANS_ELEMENT then
    begin
        k := 0;
        for j := 1 to Nterms do
        begin
            for i := 1 to Nphases do
            begin
                Inc(k);
                n := NodeRef[k];
                if n <= 0 then
                    continue;

                Result += NodeV[n] * cong(Iterminal[k]);
            end;
            Inc(k, Nphases)
        end;
    end
    else  // for all other elements
    begin
        for k := 1 to Yorder do
        begin
            n := NodeRef[k];
            if n <= 0 then
                continue;

            Result += NodeV[n] * cong(Iterminal[k]);
        end;
    end;

    if ActiveCircuit.PositiveSequence then
        Result *= 3.0;
end;

function TDSSCktElement.Get_MaxVoltageC(idxTerm: Integer): Complex;
// Get Voltage at the specified terminal 09/17/2019
var
    volts: Complex;
    ClassIdx,
    i, k,
    nrefN,
    nref: Integer;
    MaxCurr,
    CurrMag: Double;
    MaxPhase: Integer;
    NodeV: pNodeVarray;
begin
    ActiveTerminalIdx := idxTerm;   // set active Terminal
    Result := 0;
    if (not FEnabled) or (NodeRef = NIL) then
        Exit;
        
    ComputeIterminal();

    // Method: Checks what's the phase with maximum current
    // retunrs the voltage for that phase

    MaxCurr := 0.0;
    MaxPhase := 1;  // Init this so it has a non zero value
    k := (idxTerm - 1) * Fnconds; // starting index of terminal
    for i := 1 to Fnphases do
    begin
        CurrMag := Cabs(Iterminal[k + i]);
        if CurrMag > MaxCurr then
        begin
            MaxCurr := CurrMag;
            MaxPhase := i
        end;
    end;

    NodeV := ActiveCircuit.Solution.NodeV;
    ClassIdx := DSSObjType and CLASSMASK;              // gets the parent class descriptor (int)
    nref := ActiveTerminal^.TermNodeRef[MaxPhase - 1]; // reference to the phase voltage with the max current
    nrefN := ActiveTerminal^.TermNodeRef[Fnconds - 1];  // reference to the ground terminal (GND or other phase)
    // Get power into max phase of active terminal
    if not (ClassIdx = XFMR_ELEMENT) then  // Only for transformers
        volts := NodeV[nref]
    else
        volts := NodeV[nref] - NodeV[nrefN];
    Result := volts;
end;

function TDSSCktElement.Get_MaxVoltage(idxTerm: Integer): double;
begin
    Result := cabs(Get_MaxVoltageC(idxTerm));
end;

function TDSSCktElement.Get_MaxVoltageAng(idxTerm: Integer): double;
begin
    Result := cang(Get_MaxVoltageC(idxTerm));
end;

function TDSSCktElement.Get_MaxPower(idxTerm: Integer): Complex;
//Get power in the phase with the max current and return equivalent power as if it were balanced in all phases
// 2/12/2019
var
    volts: Complex;
    ClassIdx,
    i, k,
    nrefN,
    nref: Integer;
    MaxCurr,
    CurrMag: Double;
    MaxPhase: Integer;
    NodeV: pNodeVarray;
begin
    ActiveTerminalIdx := idxTerm;   // set active Terminal
    Result := 0;
    if (not FEnabled) or (NodeRef = NIL) then
        Exit;
        
    ComputeIterminal();

    // Method: Get power in the phase with max current of active terminal
    // Multiply by Nphases and return

    MaxCurr := 0.0;
    MaxPhase := 1;  // Init this so it has a non zero value
    k := (idxTerm - 1) * Fnconds; // starting index of terminal
    for i := 1 to Fnphases do
    begin
        CurrMag := Cabs(Iterminal[k + i]);
        if CurrMag > MaxCurr then
        begin
            MaxCurr := CurrMag;
            MaxPhase := i
        end;
    end;

    NodeV := ActiveCircuit.Solution.NodeV;
    ClassIdx := DSSObjType and CLASSMASK;              // gets the parent class descriptor (int)
    nref := ActiveTerminal^.TermNodeRef[MaxPhase - 1]; // reference to the phase voltage with the max current
    nrefN := ActiveTerminal^.TermNodeRef[Fnconds - 1];  // reference to the ground terminal (GND or other phase)
    
    // Get power into max phase of active terminal
    if not (ClassIdx = XFMR_ELEMENT) then  // Only for transformers
        volts := NodeV[nref]
    else
        volts := NodeV[nref] - NodeV[nrefN];
    Result := volts * cong(Iterminal[k + MaxPhase]);

    // Compute equivalent total power of all phases assuming equal to max power in all phases
    Result := Result * Fnphases;

    // If this is a positive sequence circuit (Fnphases=1),
    // then we need to multiply by 3 to get the 3-phase power
    if ActiveCircuit.PositiveSequence then
        Result := Result * 3.0;
end;

function TDSSCktElement.Get_MaxCurrent(idxTerm: Integer): Double;
// returns the magnitude fo the maximum current at the element's terminal
var
    i, k: Integer;
    CurrMag: Double;
    // MaxPhase: Integer;
begin
    ActiveTerminalIdx := idxTerm;   // set active Terminal
    Result := 0.0;
    if (not FEnabled) or (NodeRef = NIL) then
        Exit;
        
    ComputeIterminal;
    // Method: Get max current at terminal (magnitude)
    // MaxPhase := 1;  // Init this so it has a non zero value
    k := (idxTerm - 1) * Fnconds; // starting index of terminal
    for i := 1 to Fnphases do
    begin
        CurrMag := Cabs(Iterminal[k + i]);
        if CurrMag > Result then
        begin
            Result := CurrMag;
            // MaxPhase := i
        end;
    end;
end;

function TDSSCktElement.Get_MaxCurrentAng(idxTerm: Integer): Double;
// returns the angle fo the maximum current at the element's terminal
var
    i, k: Integer;
    CurrAng,
    MaxCurr,
    CurrMag: Double;
    // nref: Integer;
    // MaxPhase: Integer;
begin
    ActiveTerminalIdx := idxTerm;   // set active Terminal
    Result := 0.0;
    if (not FEnabled) or (NodeRef = NIL) then
        Exit;

    CurrAng := 0.0;
    ComputeIterminal();
    // Method: Get max current at terminal (magnitude)
    MaxCurr := 0.0;
    // MaxPhase := 1;  // Init this so it has a non zero value
    k := (idxTerm - 1) * Fnconds; // starting index of terminal
    for i := 1 to Fnphases do
    begin
        CurrMag := Cabs(Iterminal[k + i]);
        if CurrMag > MaxCurr then
        begin
            MaxCurr := CurrMag;
            CurrAng := Cang(Iterminal[k + i]);
            // MaxPhase := i
        end;
    end;
    Result := CurrAng;
end;

function TDSSCktElement.Get_PCE_Value(idxTerm: Integer; ValType: Integer): Double;
begin
    case ValType of
        0, 7:
            Result := -Power[1].re;             // P, P0
        1, 8:
            Result := -Power[1].im;             // Q, Q0
        2:
            Result := MaxVoltage[1];             // VMag
        3:
            Result := MaxVoltageAng[1];          // VAng
        4:
            Result := MaxCurrent[1];             // IMag
        5:
            Result := MaxCurrentAng[1];          // IAng
        6:
            Result := cabs(Power[1]);            // S
    else
        Result := 0;
    end;
end;

procedure TDSSCktElement.GetPhasePower(PowerBuffer: pComplexArray);
// Get the power in each phase (complex losses) of active terminal
// neutral conductors are ignored by this routine
var
    i, n: Integer;
    NodeV: pNodeVarray;
begin
    if (not FEnabled) or (NodeRef = NIL) then
    begin
        FillByte(PowerBuffer^, Yorder * (SizeOf(Double) * 2), 0);
        Exit;
    end;
    
    ComputeIterminal();

    NodeV := ActiveCircuit.Solution.NodeV;
    for i := 1 to Yorder do
    begin
        n := NodeRef[i]; // increment through terminals
        if n > 0 then
        begin
            if ActiveCircuit.PositiveSequence then
                PowerBuffer[i] := NodeV[n] * cong(Iterminal[i]) * 3.0
            else
                PowerBuffer[i] := NodeV[n] * cong(Iterminal[i]);
        end;
    end;
end;

procedure TDSSCktElement.GetPhaseLosses(var Num_Phases: Integer; LossBuffer: pComplexArray);
// Get the losses in each phase (complex losses);  Power difference coming out
// each phase. Note: This can be misleading if the nodev voltage is greatly unbalanced.
// neutral conductors are ignored by this routine
var
    i, j, k, n: Integer;
    cLoss: Complex;
    NodeV: pNodeVarray;
begin
    Num_Phases := Fnphases;

    if (not FEnabled) or (NodeRef = NIL) then
    begin
        FillByte(LossBuffer^, Fnphases * (SizeOf(Double) * 2), 0);
        Exit;
    end;
    
    ComputeIterminal();

    NodeV := ActiveCircuit.Solution.NodeV;
    for i := 1 to Num_Phases do
    begin
        cLoss := 0;
        for j := 1 to FNTerms do
        begin
            k := (j - 1) * FNconds + i;
            n := NodeRef[k]; // increment through terminals
            if n > 0 then
            begin
                if ActiveCircuit.PositiveSequence then
                    cLoss += NodeV[n] * cong(Iterminal[k]) * 3.0
                else
                    cLoss += NodeV[n] * cong(Iterminal[k]);
            end;
        end;
        LossBuffer[i] := cLoss;
    end;
end;

procedure TDSSCktElement.DumpProperties(F: TStream; Complete: Boolean; Leaf: Boolean);
var
    i, j: Integer;
begin
    inherited DumpProperties(F, Complete, Leaf);

    if FEnabled then
        FSWriteln(F, '! ENABLED')
    else
        FSWriteln(F, '! DISABLED');
    if Complete then
    begin
        FSWriteln(F, '! NPhases = ', IntToStr(Fnphases));
        FSWriteln(F, '! Nconds = ', IntToStr(Fnconds));
        FSWriteln(F, '! Nterms = ', IntToStr(fNterms));
        FSWriteln(F, '! Yorder = ', IntToStr(Yorder));
        FSWrite(F, '! NodeRef = "');
        if NodeRef = NIL then
            FSWrite(F, 'nil')
        else
            for i := 1 to Yorder do
                FSWrite(F, IntToStr(NodeRef[i]), ' ');
        FSWriteln(F, '"');
        FSWrite(F, '! Terminal Status: [');
        for i := 1 to fNTerms do
            for j := 1 to Fnconds do
            begin
                if Terminals[i - 1].ConductorsClosed[j - 1] then
                    FSWrite(F, 'C ')
                else
                    FSWrite(F, 'O ');
            end;
        FSWriteln(F, ']');
        FSWrite(F, '! Terminal Bus Ref: [');
        for i := 1 to fNTerms do
            for j := 1 to Fnconds do
            begin
                FSWrite(F, IntToStr(Terminals[i - 1].BusRef), ' ');
            end;
        FSWriteln(F, ']');
        FSWriteln(F);

        if YPrim <> NIL then
        begin
            FSWriteln(F, '! YPrim (G matrix)');
            for i := 1 to Yorder do
            begin
                FSWrite(F, '! ');
                for j := 1 to Yorder do
                    FSWrite(F, Format(' %13.10g |', [YPrim[i, j].re]));
                FSWriteln(F);
            end;
            FSWriteln(F, '! YPrim (B Matrix) = ');
            for i := 1 to Yorder do
            begin
                FSWrite(F, '! ');
                for j := 1 to Yorder do
                    FSWrite(F, Format(' %13.10g |', [YPrim[i, j].im]));
                FSWriteln(F);
            end;
        end;
    end;
end;

procedure TDSSCktElement.DoYprimCalcs(Ymatrix: TCMatrix);
var
    i, j, k, ii, jj, ElimRow: Integer;
    Ynn, Yij, Yin, Ynj: Complex;
    RowEliminated: pIntegerArray;
    ElementOpen: Boolean;
begin
    // Now Account for Open Conductors
    // Perform a Kron Reduction on rows where I is forced to zero.
    // Then for any conductor that is open, zero out row and column.
    ElementOpen := FALSE;
    k := 0;
    for i := 1 to fNTerms do
    begin
        for j := 1 to Fnconds do
        begin
            if not Terminals[i - 1].ConductorsClosed[j - 1] then
            begin
                if not ElementOpen then
                begin
                    RowEliminated := AllocMem(Sizeof(Integer) * Yorder);
                    ElementOpen := TRUE;
                end;
                // First do Kron Reduction
                ElimRow := j + k;
                Ynn := Ymatrix[ElimRow, ElimRow];
                if Cabs(Ynn) = 0.0 then
                    Ynn.re := EPSILON;
                RowEliminated[ElimRow] := 1;
                for ii := 1 to Yorder do
                begin
                    if RowEliminated[ii] = 0 then
                    begin
                        Yin := Ymatrix[ii, ElimRow];
                        for jj := ii to Yorder do
                            if RowEliminated[jj] = 0 then
                            begin
                                Yij := Ymatrix[ii, jj];
                                Ynj := Ymatrix[ElimRow, jj];
                                Ymatrix[ii, jj] := Yij - ((Yin * Ynj) / Ynn);
                                Ymatrix[jj, ii] := Ymatrix[ii, jj];
                            end;
                    end;
                end;
                // Now zero out row and column
                Ymatrix.ZeroRow(ElimRow);
                Ymatrix.ZeroCol(ElimRow);
                // put a small amount on the diagonal in case node gets isolated
                Ymatrix[ElimRow, ElimRow] := cEpsilon;
            end;
        end;
        k := k + Fnconds;
    end;
    // Clean up at end of loop.
    // Add in cEpsilon to diagonal elements of remaining rows to avoid leaving a bus hanging.
    // This happens on low-impedance simple from-to elements when one terminal opened.
    if ElementOpen then
    begin
        for ii := 1 to Yorder do
            if RowEliminated[ii] = 0 then
                Ymatrix.AddElement(ii, ii, cEpsilon);

        Reallocmem(RowEliminated, 0);
    end;
end;

procedure TDSSCktElement.SumCurrents;
// sum Terminal Currents into System  Currents Array
// Primarily for Newton Iteration
var
    i: Integer;
    Currents: pNodeVArray;
begin
    if (not FEnabled) or (NodeRef = NIL) then
        Exit;
        
    ComputeIterminal();
    Currents := ActiveCircuit.Solution.Currents;
    for i := 1 to Yorder do
        Currents[NodeRef[i]] += Iterminal[i];  // Noderef=0 is OK
end;

procedure TDSSCktElement.GetTermVoltages(iTerm: Integer; VBuffer: PComplexArray);
// Bus Voltages at indicated terminal
// Fill Vbuffer array which must be adequately allocated by calling routine
var
    ncond, i: Integer;
    NodeV: pNodeVarray;
begin
    try
        ncond := NConds;

        // return Zero if terminal number improperly specified
        if (iTerm < 1) or (iTerm > fNterms) then
        begin
            for i := 1 to Ncond do
                VBuffer[i] := 0;
            Exit;
        end;

        NodeV := ActiveCircuit.Solution.NodeV;
        for i := 1 to NCond do
            Vbuffer[i] := NodeV[Terminals[iTerm - 1].TermNodeRef[i - 1]];

    except
        On E: Exception do
            DoSimpleMsg('Error filling voltage buffer in GetTermVoltages for Circuit Element: "%s". Probable Cause: Invalid definition of element. System Error Message: %s', [FullName, E.Message], 755);
    end;
end;

procedure TDSSCktElement.GetSeqLosses(var PosSeqLosses, NegSeqLosses, ZeroModeLosses: complex);
begin
// For the base class, just return 0
// Derived classes have to supply appropriate function
    PosSeqLosses := 0;
    NegSeqLosses := 0;
    ZeroModeLosses := 0;
end;

function IsGroundBus(const S: String): Boolean;
var
    i: Integer;
begin
    Result := TRUE;
    i := pos('.1', S);
    if i > 0 then
        Result := FALSE;
    i := pos('.2', S);
    if i > 0 then
        Result := FALSE;
    i := pos('.3', S);
    if i > 0 then
        Result := FALSE;
    i := pos('.', S);
    if i = 0 then
        Result := FALSE;
end;

procedure TDSSCktElement.MakePosSequence();
var
    i: Integer;
    grnd: Boolean;
begin
    for i := 1 to FNterms do
    begin
        grnd := IsGroundBus(FBusNames[i]);
        FBusNames[i] := StripExtension(FBusNames[i]);
        if grnd then
            FBusNames[i] := FBusNames[i] + '.0';
    end;
end;

procedure TDSSCktElement.ComputeVterminal;
// Put terminal voltages in an array
var
    i: Integer;
    vterm: PDouble;
    nref: PInteger;
    nv0, nv: PDouble;
begin
    vterm := PDouble(VTerminal);
    nref := PInteger(NodeRef);
    nv0 := PDouble(ActiveCircuit.solution.NodeV);
    for i := 1 to Yorder do
    begin
        nv := nv0 + 2 * nref^;
        vterm^ := nv^;
        (vterm + 1)^ := (nv + 1)^;
        inc(vterm, 2);
        inc(nref);
        // VTerminal[i] := NodeV[NodeRef[i]];
    end;
end;

procedure TDSSCktElement.ZeroITerminal; inline;
var
    i: Integer;
    it: PDouble;
begin
    // Somehow this is slower?! FillDWord(ITerminal^, Yorder * ((SizeOf(Double) * 2) div 4), 0);
    it := PDouble(Iterminal);
    for i := 1 to Yorder do
    begin
        it^ := 0;
        (it + 1)^ := 0;
        inc(it, 2);
    end;
    //for i := 1 to Yorder do
    //    ITerminal[i] := 0;
end;

procedure TDSSCktElement.MakeLike(OtherObj: Pointer);
var
    OtherCktObj: TDSSCktElement;
begin
    inherited MakeLike(OtherObj);

    OtherCktObj := TDSSCktElement(OtherObj);
    BaseFrequency := OtherCktObj.BaseFrequency;
    Enabled := TRUE;
end;

function TDSSCktElement.AllConductorsClosed(): Boolean;
var
    i, j: Integer;
begin
    Result := TRUE;
    for i := 1 to Nterms do
        for j := 1 to NConds do
            if not Terminals[i - 1].ConductorsClosed[j - 1] then
            begin
                Result := FALSE;
                Exit;
            end;
end;

procedure TDSSCktElement.GetCurrents(Curr: ArrayOfComplex);
begin
    GetCurrents(pComplexArray(@Curr[0]));
end;

end.
