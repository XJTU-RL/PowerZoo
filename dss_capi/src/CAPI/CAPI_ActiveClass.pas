unit CAPI_ActiveClass;

interface

uses
    CAPI_Utils,
    CAPI_Types;

procedure ActiveClass_Get_AllNames(var ResultPtr: PPAnsiChar; ResultCount: PAPISize); CDECL;
procedure ActiveClass_Get_AllNames_GR(); CDECL;
function ActiveClass_Get_First(): Integer; CDECL;
function ActiveClass_Get_Next(): Integer; CDECL;
function ActiveClass_Get_Name(): PAnsiChar; CDECL;
procedure ActiveClass_Set_Name(const Value: PAnsiChar); CDECL;
function ActiveClass_Get_NumElements(): Integer; CDECL;
function ActiveClass_Get_ActiveClassName(): PAnsiChar; CDECL;
function ActiveClass_Get_Count(): Integer; CDECL;
function ActiveClass_Get_ActiveClassParent(): PAnsiChar; CDECL;

// API extensions
function ActiveClass_ToJSON(joptions: Integer): PAnsiChar; CDECL;
function ActiveClass_Get_Pointer(): Pointer; CDECL;

implementation

uses
    CAPI_Constants,
    DSSGlobals,
    DSSObject,
    CktElement,
    PCClass, 
    PDClass, 
    DSSClass,
    DSSHelper,
    MeterClass, 
    ControlClass,
    CAPI_Obj,
    DSSObjectHelper,
    fpjson,
    sysutils;

procedure ActiveClass_Get_AllNames(var ResultPtr: PPAnsiChar; ResultCount: PAPISize); CDECL;
var
    Result: PPAnsiCharArray0;
    k: Integer;
    elem: TDSSObject;
begin
    if (InvalidCircuit(DSSPrime)) or (DSSPrime.ActiveDSSClass = NIL) then
    begin
        DefaultResult(ResultPtr, ResultCount, '');
        Exit;
    end;
        
    Result := DSS_RecreateArray_PPAnsiChar(ResultPtr, ResultCount, DSSPrime.ActiveDSSClass.ElementCount());
    k := 0;
    for elem in DSSPrime.ActiveDSSClass do
    begin
        Result[k] := DSS_CopyStringAsPChar(elem.Name);
        k += 1;
    end;
end;

procedure ActiveClass_Get_AllNames_GR(); CDECL;
// Same as ActiveClass_Get_AllNames but uses global result (GR) pointers
begin
    ActiveClass_Get_AllNames(DSSPrime.GR_DataPtr_PPAnsiChar, @DSSPrime.GR_Counts_PPAnsiChar[0])
end;

//------------------------------------------------------------------------------
function ActiveClass_Get_First(): Integer; CDECL;
begin
    Result := 0;
    if (InvalidCircuit(DSSPrime)) or (DSSPrime.ActiveDSSClass = NIL) then
        Exit;
    Result := DSSPrime.ActiveDSSClass.First();  // sets active objects
end;
//------------------------------------------------------------------------------
function ActiveClass_Get_Next(): Integer; CDECL;
begin
    Result := 0;
    if (InvalidCircuit(DSSPrime)) or (DSSPrime.ActiveDSSClass = NIL) then
        Exit;
    Result := DSSPrime.ActiveDSSClass.Next();  // sets active objects
end;
//------------------------------------------------------------------------------
function ActiveClass_Get_Name(): PAnsiChar; CDECL;
begin
    Result := NIL;
    if DSSPrime.ActiveDSSObject = NIL then
        Exit;
    
    Result := DSS_GetAsPAnsiChar(DSSPrime, DSSPrime.ActiveDSSObject.Name)
end;
//------------------------------------------------------------------------------
procedure ActiveClass_Set_Name(const Value: PAnsiChar); CDECL;
// set object active by name
var
    pelem: TDSSObject;
begin
    if DSSPrime.ActiveDSSClass = NIL then
        Exit;
        
    pelem := DSSPrime.ActiveDSSClass.Find(Value);
    if pelem = NIL then
        Exit;

    if pelem is TDSSCktElement then
        DSSPrime.ActiveCircuit.ActiveCktElement := TDSSCktElement(pelem)  // sets DSSPrime.ActiveDSSObject
    else
        DSSPrime.ActiveDSSObject := pelem;
end;
//------------------------------------------------------------------------------
function ActiveClass_Get_NumElements(): Integer; CDECL;
begin
    Result := 0;
    if DSSPrime.ActiveDSSClass = NIL then
        Exit;
    Result := DSSPrime.ActiveDSSCLass.ElementCount()
end;
//------------------------------------------------------------------------------
function ActiveClass_Get_ActiveClassName(): PAnsiChar; CDECL;
begin
    Result := NIL;
    if DSSPrime.ActiveDSSClass = NIL then
        Exit;
    Result := DSS_GetAsPAnsiChar(DSSPrime, DSSPrime.ActiveDSSCLass.Name)
end;
//------------------------------------------------------------------------------
function ActiveClass_Get_Count(): Integer; CDECL;
begin
    Result := 0;
    if DSSPrime.ActiveDSSClass = NIL then
        Exit;
    Result := DSSPrime.ActiveDSSCLass.ElementCount()
end;
//------------------------------------------------------------------------------
function ActiveClass_Get_ActiveClassParent(): PAnsiChar; CDECL;
begin
    if DSSPrime.ActiveDSSClass = NIL then
    begin
        Result := DSS_GetAsPAnsiChar(DSSPrime, 'Parent Class unknown');
        Exit;
    end;

    if DSSPrime.ActiveDSSClass.ClassType.InheritsFrom(TMeterClass) then
        Result := DSS_GetAsPAnsiChar(DSSPrime, 'TMeterClass')
    else if DSSPrime.ActiveDSSClass.ClassType.InheritsFrom(TControlClass) then
        Result := DSS_GetAsPAnsiChar(DSSPrime, 'TControlClass')
    else  if DSSPrime.ActiveDSSClass.ClassType.InheritsFrom(TPDClass) then
        Result := DSS_GetAsPAnsiChar(DSSPrime, 'TPDClass')
    else if DSSPrime.ActiveDSSClass.ClassType.InheritsFrom(TPCClass) then
        Result := DSS_GetAsPAnsiChar(DSSPrime, 'TPCClass')
    else 
        Result := DSS_GetAsPAnsiChar(DSSPrime, 'Generic Object');
end;
//------------------------------------------------------------------------------
function ActiveClass_ToJSON(joptions: Integer): PAnsiChar; CDECL;
var
    json: TJSONArray = NIL;
    cls: TDSSClass = NIL;
    objlist: TDSSObjectPtr = NIL;
    i: Integer;
begin
    Result := NIL;
    if (InvalidCircuit(DSSPrime)) or (DSSPrime.ActiveDSSClass = NIL) then
        Exit;

    try
        json := TJSONArray.Create([]);
        cls := DSSPrime.ActiveDSSClass;
        objlist := TDSSObjectPtr(cls.ElementList.InternalPointer);    
        if cls.ElementList.Count <> 0 then
        begin
            if ((joptions and Integer(DSSJSONOptions.ExcludeDisabled)) = 0) or not (objlist^ is TDSSCktElement) then
            begin
                for i := 1 to cls.ElementList.Count do 
                begin
                    json.Add(Obj_ToJSONData(objlist^, joptions));
                    inc(objlist);
                end;
            end
            else
            begin
                for i := 1 to cls.ElementList.Count do 
                begin
                    if TDSSCktElement(objlist^).Enabled then
                        json.Add(Obj_ToJSONData(objlist^, joptions));
                    inc(objlist);
                end;
            end;
        end;
        if json <> NIL then
        begin
            if (Integer(DSSJSONOptions.Pretty) and joptions) <> 0 then
                Result := DSS_GetAsPAnsiChar(DSSPrime, json.FormatJSON([], 2))
            else
                Result := DSS_GetAsPAnsiChar(DSSPrime, json.FormatJSON([foSingleLineArray, foSingleLineObject, foskipWhiteSpace], 0));
        end;
    except 
    on E: Exception do
        DoSimpleMsg(DSSPrime, 'Error converting to JSON: %s', [E.Message], 20231030);
    end;
    FreeAndNil(json);
end;
//------------------------------------------------------------------------------
function ActiveClass_Get_Pointer(): Pointer; CDECL;
begin
    Result := DSSPrime.ActiveDSSObject
end;
//------------------------------------------------------------------------------
end.
