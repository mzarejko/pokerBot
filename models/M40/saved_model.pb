ó

§
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12unknown8ÿû
~
dense_344/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namedense_344/kernel
w
$dense_344/kernel/Read/ReadVariableOpReadVariableOpdense_344/kernel* 
_output_shapes
:
*
dtype0
u
dense_344/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_344/bias
n
"dense_344/bias/Read/ReadVariableOpReadVariableOpdense_344/bias*
_output_shapes	
:*
dtype0
~
dense_345/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namedense_345/kernel
w
$dense_345/kernel/Read/ReadVariableOpReadVariableOpdense_345/kernel* 
_output_shapes
:
*
dtype0
u
dense_345/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_345/bias
n
"dense_345/bias/Read/ReadVariableOpReadVariableOpdense_345/bias*
_output_shapes	
:*
dtype0
~
dense_346/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namedense_346/kernel
w
$dense_346/kernel/Read/ReadVariableOpReadVariableOpdense_346/kernel* 
_output_shapes
:
*
dtype0
u
dense_346/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_346/bias
n
"dense_346/bias/Read/ReadVariableOpReadVariableOpdense_346/bias*
_output_shapes	
:*
dtype0

batch_normalization_86/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_86/gamma

0batch_normalization_86/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_86/gamma*
_output_shapes	
:*
dtype0

batch_normalization_86/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_86/beta

/batch_normalization_86/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_86/beta*
_output_shapes	
:*
dtype0

"batch_normalization_86/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_86/moving_mean

6batch_normalization_86/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_86/moving_mean*
_output_shapes	
:*
dtype0
¥
&batch_normalization_86/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_86/moving_variance

:batch_normalization_86/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_86/moving_variance*
_output_shapes	
:*
dtype0
}
dense_347/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*!
shared_namedense_347/kernel
v
$dense_347/kernel/Read/ReadVariableOpReadVariableOpdense_347/kernel*
_output_shapes
:	*
dtype0
t
dense_347/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_347/bias
m
"dense_347/bias/Read/ReadVariableOpReadVariableOpdense_347/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/dense_344/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_344/kernel/m

+Adam/dense_344/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_344/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_344/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_344/bias/m
|
)Adam/dense_344/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_344/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_345/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_345/kernel/m

+Adam/dense_345/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_345/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_345/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_345/bias/m
|
)Adam/dense_345/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_345/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_346/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_346/kernel/m

+Adam/dense_346/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_346/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_346/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_346/bias/m
|
)Adam/dense_346/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_346/bias/m*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_86/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_86/gamma/m

7Adam/batch_normalization_86/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_86/gamma/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_86/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_86/beta/m

6Adam/batch_normalization_86/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_86/beta/m*
_output_shapes	
:*
dtype0

Adam/dense_347/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdam/dense_347/kernel/m

+Adam/dense_347/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_347/kernel/m*
_output_shapes
:	*
dtype0

Adam/dense_347/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_347/bias/m
{
)Adam/dense_347/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_347/bias/m*
_output_shapes
:*
dtype0

Adam/dense_344/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_344/kernel/v

+Adam/dense_344/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_344/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_344/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_344/bias/v
|
)Adam/dense_344/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_344/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_345/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_345/kernel/v

+Adam/dense_345/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_345/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_345/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_345/bias/v
|
)Adam/dense_345/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_345/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_346/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_346/kernel/v

+Adam/dense_346/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_346/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_346/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_346/bias/v
|
)Adam/dense_346/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_346/bias/v*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_86/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_86/gamma/v

7Adam/batch_normalization_86/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_86/gamma/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_86/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_86/beta/v

6Adam/batch_normalization_86/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_86/beta/v*
_output_shapes	
:*
dtype0

Adam/dense_347/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdam/dense_347/kernel/v

+Adam/dense_347/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_347/kernel/v*
_output_shapes
:	*
dtype0

Adam/dense_347/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_347/bias/v
{
)Adam/dense_347/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_347/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Ô:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*:
value:B: Bû9
Î
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	optimizer
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
 
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api

$axis
	%gamma
&beta
'moving_mean
(moving_variance
)regularization_losses
*	variables
+trainable_variables
,	keras_api
h

-kernel
.bias
/regularization_losses
0	variables
1trainable_variables
2	keras_api
ô
3iter

4beta_1

5beta_2
	6decay
7learning_ratem`mambmcmdme%mf&mg-mh.mivjvkvlvmvnvo%vp&vq-vr.vs
 
V
0
1
2
3
4
5
%6
&7
'8
(9
-10
.11
F
0
1
2
3
4
5
%6
&7
-8
.9
­

8layers
9metrics
	regularization_losses
:non_trainable_variables

	variables
;layer_regularization_losses
<layer_metrics
trainable_variables
 
 
 
 
­
=metrics

>layers
regularization_losses
?non_trainable_variables
	variables
@layer_regularization_losses
Alayer_metrics
trainable_variables
\Z
VARIABLE_VALUEdense_344/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_344/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
Bmetrics

Clayers
regularization_losses
Dnon_trainable_variables
	variables
Elayer_regularization_losses
Flayer_metrics
trainable_variables
\Z
VARIABLE_VALUEdense_345/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_345/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
Gmetrics

Hlayers
regularization_losses
Inon_trainable_variables
	variables
Jlayer_regularization_losses
Klayer_metrics
trainable_variables
\Z
VARIABLE_VALUEdense_346/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_346/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
Lmetrics

Mlayers
 regularization_losses
Nnon_trainable_variables
!	variables
Olayer_regularization_losses
Player_metrics
"trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_86/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_86/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_86/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_86/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

%0
&1
'2
(3

%0
&1
­
Qmetrics

Rlayers
)regularization_losses
Snon_trainable_variables
*	variables
Tlayer_regularization_losses
Ulayer_metrics
+trainable_variables
\Z
VARIABLE_VALUEdense_347/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_347/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

-0
.1

-0
.1
­
Vmetrics

Wlayers
/regularization_losses
Xnon_trainable_variables
0	variables
Ylayer_regularization_losses
Zlayer_metrics
1trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
1
0
1
2
3
4
5
6

[0

'0
(1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

'0
(1
 
 
 
 
 
 
 
4
	\total
	]count
^	variables
_	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

\0
]1

^	variables
}
VARIABLE_VALUEAdam/dense_344/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_344/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_345/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_345/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_346/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_346/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_86/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_86/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_347/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_347/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_344/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_344/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_345/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_345/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_346/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_346/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_86/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_86/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_347/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_347/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_87Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ4
Þ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_87dense_344/kerneldense_344/biasdense_345/kerneldense_345/biasdense_346/kerneldense_346/bias&batch_normalization_86/moving_variancebatch_normalization_86/gamma"batch_normalization_86/moving_meanbatch_normalization_86/betadense_347/kerneldense_347/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *1
f,R*
(__inference_signature_wrapper_1128688519
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Æ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_344/kernel/Read/ReadVariableOp"dense_344/bias/Read/ReadVariableOp$dense_345/kernel/Read/ReadVariableOp"dense_345/bias/Read/ReadVariableOp$dense_346/kernel/Read/ReadVariableOp"dense_346/bias/Read/ReadVariableOp0batch_normalization_86/gamma/Read/ReadVariableOp/batch_normalization_86/beta/Read/ReadVariableOp6batch_normalization_86/moving_mean/Read/ReadVariableOp:batch_normalization_86/moving_variance/Read/ReadVariableOp$dense_347/kernel/Read/ReadVariableOp"dense_347/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_344/kernel/m/Read/ReadVariableOp)Adam/dense_344/bias/m/Read/ReadVariableOp+Adam/dense_345/kernel/m/Read/ReadVariableOp)Adam/dense_345/bias/m/Read/ReadVariableOp+Adam/dense_346/kernel/m/Read/ReadVariableOp)Adam/dense_346/bias/m/Read/ReadVariableOp7Adam/batch_normalization_86/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_86/beta/m/Read/ReadVariableOp+Adam/dense_347/kernel/m/Read/ReadVariableOp)Adam/dense_347/bias/m/Read/ReadVariableOp+Adam/dense_344/kernel/v/Read/ReadVariableOp)Adam/dense_344/bias/v/Read/ReadVariableOp+Adam/dense_345/kernel/v/Read/ReadVariableOp)Adam/dense_345/bias/v/Read/ReadVariableOp+Adam/dense_346/kernel/v/Read/ReadVariableOp)Adam/dense_346/bias/v/Read/ReadVariableOp7Adam/batch_normalization_86/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_86/beta/v/Read/ReadVariableOp+Adam/dense_347/kernel/v/Read/ReadVariableOp)Adam/dense_347/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_save_1128689006
µ	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_344/kerneldense_344/biasdense_345/kerneldense_345/biasdense_346/kerneldense_346/biasbatch_normalization_86/gammabatch_normalization_86/beta"batch_normalization_86/moving_mean&batch_normalization_86/moving_variancedense_347/kerneldense_347/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_344/kernel/mAdam/dense_344/bias/mAdam/dense_345/kernel/mAdam/dense_345/bias/mAdam/dense_346/kernel/mAdam/dense_346/bias/m#Adam/batch_normalization_86/gamma/m"Adam/batch_normalization_86/beta/mAdam/dense_347/kernel/mAdam/dense_347/bias/mAdam/dense_344/kernel/vAdam/dense_344/bias/vAdam/dense_345/kernel/vAdam/dense_345/bias/vAdam/dense_346/kernel/vAdam/dense_346/bias/v#Adam/batch_normalization_86/gamma/v"Adam/batch_normalization_86/beta/vAdam/dense_347/kernel/vAdam/dense_347/bias/v*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference__traced_restore_1128689133ÇÒ
ë

.__inference_dense_345_layer_call_fn_1128688744

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_345_layer_call_and_return_conditional_losses_11286882132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
"

H__inference_model_86_layer_call_and_return_conditional_losses_1128688390

inputs
dense_344_1128688360
dense_344_1128688362
dense_345_1128688365
dense_345_1128688367
dense_346_1128688370
dense_346_1128688372%
!batch_normalization_86_1128688375%
!batch_normalization_86_1128688377%
!batch_normalization_86_1128688379%
!batch_normalization_86_1128688381
dense_347_1128688384
dense_347_1128688386
identity¢.batch_normalization_86/StatefulPartitionedCall¢!dense_344/StatefulPartitionedCall¢!dense_345/StatefulPartitionedCall¢!dense_346/StatefulPartitionedCall¢!dense_347/StatefulPartitionedCallß
flatten_86/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_flatten_86_layer_call_and_return_conditional_losses_11286881672
flatten_86/PartitionedCallÃ
!dense_344/StatefulPartitionedCallStatefulPartitionedCall#flatten_86/PartitionedCall:output:0dense_344_1128688360dense_344_1128688362*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_344_layer_call_and_return_conditional_losses_11286881862#
!dense_344/StatefulPartitionedCallÊ
!dense_345/StatefulPartitionedCallStatefulPartitionedCall*dense_344/StatefulPartitionedCall:output:0dense_345_1128688365dense_345_1128688367*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_345_layer_call_and_return_conditional_losses_11286882132#
!dense_345/StatefulPartitionedCallÊ
!dense_346/StatefulPartitionedCallStatefulPartitionedCall*dense_345/StatefulPartitionedCall:output:0dense_346_1128688370dense_346_1128688372*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_346_layer_call_and_return_conditional_losses_11286882402#
!dense_346/StatefulPartitionedCallÓ
.batch_normalization_86/StatefulPartitionedCallStatefulPartitionedCall*dense_346/StatefulPartitionedCall:output:0!batch_normalization_86_1128688375!batch_normalization_86_1128688377!batch_normalization_86_1128688379!batch_normalization_86_1128688381*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_batch_normalization_86_layer_call_and_return_conditional_losses_112868811320
.batch_normalization_86/StatefulPartitionedCallÖ
!dense_347/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_86/StatefulPartitionedCall:output:0dense_347_1128688384dense_347_1128688386*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_347_layer_call_and_return_conditional_losses_11286883022#
!dense_347/StatefulPartitionedCall¿
IdentityIdentity*dense_347/StatefulPartitionedCall:output:0/^batch_normalization_86/StatefulPartitionedCall"^dense_344/StatefulPartitionedCall"^dense_345/StatefulPartitionedCall"^dense_346/StatefulPartitionedCall"^dense_347/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2`
.batch_normalization_86/StatefulPartitionedCall.batch_normalization_86/StatefulPartitionedCall2F
!dense_344/StatefulPartitionedCall!dense_344/StatefulPartitionedCall2F
!dense_345/StatefulPartitionedCall!dense_345/StatefulPartitionedCall2F
!dense_346/StatefulPartitionedCall!dense_346/StatefulPartitionedCall2F
!dense_347/StatefulPartitionedCall!dense_347/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ë

.__inference_dense_346_layer_call_fn_1128688764

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_346_layer_call_and_return_conditional_losses_11286882402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã
®
;__inference_batch_normalization_86_layer_call_fn_1128688833

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_batch_normalization_86_layer_call_and_return_conditional_losses_11286881132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é

.__inference_dense_347_layer_call_fn_1128688866

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_347_layer_call_and_return_conditional_losses_11286883022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
f
J__inference_flatten_86_layer_call_and_return_conditional_losses_1128688699

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ4:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
Ü0
Õ
V__inference_batch_normalization_86_layer_call_and_return_conditional_losses_1128688113

inputs
assignmovingavg_1128688088 
assignmovingavg_1_1128688094)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1Ð
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg/1128688088*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1128688088*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOpö
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg/1128688088*
_output_shapes	
:2
AssignMovingAvg/subí
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg/1128688088*
_output_shapes	
:2
AssignMovingAvg/mul·
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1128688088AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg/1128688088*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÖ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*/
_class%
#!loc:@AssignMovingAvg_1/1128688094*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1128688094*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*/
_class%
#!loc:@AssignMovingAvg_1/1128688094*
_output_shapes	
:2
AssignMovingAvg_1/sub÷
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*/
_class%
#!loc:@AssignMovingAvg_1/1128688094*
_output_shapes	
:2
AssignMovingAvg_1/mulÃ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1128688094AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*/
_class%
#!loc:@AssignMovingAvg_1/1128688094*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1´
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ	
â
I__inference_dense_347_layer_call_and_return_conditional_losses_1128688857

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü	
â
I__inference_dense_344_layer_call_and_return_conditional_losses_1128688715

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ	

-__inference_model_86_layer_call_fn_1128688480
input_87
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinput_87unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_model_86_layer_call_and_return_conditional_losses_11286884532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
input_87
"

H__inference_model_86_layer_call_and_return_conditional_losses_1128688453

inputs
dense_344_1128688423
dense_344_1128688425
dense_345_1128688428
dense_345_1128688430
dense_346_1128688433
dense_346_1128688435%
!batch_normalization_86_1128688438%
!batch_normalization_86_1128688440%
!batch_normalization_86_1128688442%
!batch_normalization_86_1128688444
dense_347_1128688447
dense_347_1128688449
identity¢.batch_normalization_86/StatefulPartitionedCall¢!dense_344/StatefulPartitionedCall¢!dense_345/StatefulPartitionedCall¢!dense_346/StatefulPartitionedCall¢!dense_347/StatefulPartitionedCallß
flatten_86/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_flatten_86_layer_call_and_return_conditional_losses_11286881672
flatten_86/PartitionedCallÃ
!dense_344/StatefulPartitionedCallStatefulPartitionedCall#flatten_86/PartitionedCall:output:0dense_344_1128688423dense_344_1128688425*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_344_layer_call_and_return_conditional_losses_11286881862#
!dense_344/StatefulPartitionedCallÊ
!dense_345/StatefulPartitionedCallStatefulPartitionedCall*dense_344/StatefulPartitionedCall:output:0dense_345_1128688428dense_345_1128688430*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_345_layer_call_and_return_conditional_losses_11286882132#
!dense_345/StatefulPartitionedCallÊ
!dense_346/StatefulPartitionedCallStatefulPartitionedCall*dense_345/StatefulPartitionedCall:output:0dense_346_1128688433dense_346_1128688435*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_346_layer_call_and_return_conditional_losses_11286882402#
!dense_346/StatefulPartitionedCallÕ
.batch_normalization_86/StatefulPartitionedCallStatefulPartitionedCall*dense_346/StatefulPartitionedCall:output:0!batch_normalization_86_1128688438!batch_normalization_86_1128688440!batch_normalization_86_1128688442!batch_normalization_86_1128688444*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_batch_normalization_86_layer_call_and_return_conditional_losses_112868814620
.batch_normalization_86/StatefulPartitionedCallÖ
!dense_347/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_86/StatefulPartitionedCall:output:0dense_347_1128688447dense_347_1128688449*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_347_layer_call_and_return_conditional_losses_11286883022#
!dense_347/StatefulPartitionedCall¿
IdentityIdentity*dense_347/StatefulPartitionedCall:output:0/^batch_normalization_86/StatefulPartitionedCall"^dense_344/StatefulPartitionedCall"^dense_345/StatefulPartitionedCall"^dense_346/StatefulPartitionedCall"^dense_347/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2`
.batch_normalization_86/StatefulPartitionedCall.batch_normalization_86/StatefulPartitionedCall2F
!dense_344/StatefulPartitionedCall!dense_344/StatefulPartitionedCall2F
!dense_345/StatefulPartitionedCall!dense_345/StatefulPartitionedCall2F
!dense_346/StatefulPartitionedCall!dense_346/StatefulPartitionedCall2F
!dense_347/StatefulPartitionedCall!dense_347/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
	

(__inference_signature_wrapper_1128688519
input_87
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinput_87unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference__wrapped_model_11286880172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
input_87
U

#__inference__traced_save_1128689006
file_prefix/
+savev2_dense_344_kernel_read_readvariableop-
)savev2_dense_344_bias_read_readvariableop/
+savev2_dense_345_kernel_read_readvariableop-
)savev2_dense_345_bias_read_readvariableop/
+savev2_dense_346_kernel_read_readvariableop-
)savev2_dense_346_bias_read_readvariableop;
7savev2_batch_normalization_86_gamma_read_readvariableop:
6savev2_batch_normalization_86_beta_read_readvariableopA
=savev2_batch_normalization_86_moving_mean_read_readvariableopE
Asavev2_batch_normalization_86_moving_variance_read_readvariableop/
+savev2_dense_347_kernel_read_readvariableop-
)savev2_dense_347_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_344_kernel_m_read_readvariableop4
0savev2_adam_dense_344_bias_m_read_readvariableop6
2savev2_adam_dense_345_kernel_m_read_readvariableop4
0savev2_adam_dense_345_bias_m_read_readvariableop6
2savev2_adam_dense_346_kernel_m_read_readvariableop4
0savev2_adam_dense_346_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_86_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_86_beta_m_read_readvariableop6
2savev2_adam_dense_347_kernel_m_read_readvariableop4
0savev2_adam_dense_347_bias_m_read_readvariableop6
2savev2_adam_dense_344_kernel_v_read_readvariableop4
0savev2_adam_dense_344_bias_v_read_readvariableop6
2savev2_adam_dense_345_kernel_v_read_readvariableop4
0savev2_adam_dense_345_bias_v_read_readvariableop6
2savev2_adam_dense_346_kernel_v_read_readvariableop4
0savev2_adam_dense_346_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_86_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_86_beta_v_read_readvariableop6
2savev2_adam_dense_347_kernel_v_read_readvariableop4
0savev2_adam_dense_347_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*¡
valueB(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesØ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesò
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_344_kernel_read_readvariableop)savev2_dense_344_bias_read_readvariableop+savev2_dense_345_kernel_read_readvariableop)savev2_dense_345_bias_read_readvariableop+savev2_dense_346_kernel_read_readvariableop)savev2_dense_346_bias_read_readvariableop7savev2_batch_normalization_86_gamma_read_readvariableop6savev2_batch_normalization_86_beta_read_readvariableop=savev2_batch_normalization_86_moving_mean_read_readvariableopAsavev2_batch_normalization_86_moving_variance_read_readvariableop+savev2_dense_347_kernel_read_readvariableop)savev2_dense_347_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_344_kernel_m_read_readvariableop0savev2_adam_dense_344_bias_m_read_readvariableop2savev2_adam_dense_345_kernel_m_read_readvariableop0savev2_adam_dense_345_bias_m_read_readvariableop2savev2_adam_dense_346_kernel_m_read_readvariableop0savev2_adam_dense_346_bias_m_read_readvariableop>savev2_adam_batch_normalization_86_gamma_m_read_readvariableop=savev2_adam_batch_normalization_86_beta_m_read_readvariableop2savev2_adam_dense_347_kernel_m_read_readvariableop0savev2_adam_dense_347_bias_m_read_readvariableop2savev2_adam_dense_344_kernel_v_read_readvariableop0savev2_adam_dense_344_bias_v_read_readvariableop2savev2_adam_dense_345_kernel_v_read_readvariableop0savev2_adam_dense_345_bias_v_read_readvariableop2savev2_adam_dense_346_kernel_v_read_readvariableop0savev2_adam_dense_346_bias_v_read_readvariableop>savev2_adam_batch_normalization_86_gamma_v_read_readvariableop=savev2_adam_batch_normalization_86_beta_v_read_readvariableop2savev2_adam_dense_347_kernel_v_read_readvariableop0savev2_adam_dense_347_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*½
_input_shapes«
¨: :
::
::
::::::	:: : : : : : : :
::
::
::::	::
::
::
::::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!	

_output_shapes	
::!


_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::&"
 
_output_shapes
:
:!

_output_shapes	
::& "
 
_output_shapes
:
:!!

_output_shapes	
::&""
 
_output_shapes
:
:!#

_output_shapes	
::!$

_output_shapes	
::!%

_output_shapes	
::%&!

_output_shapes
:	: '

_output_shapes
::(

_output_shapes
: 
ü	
â
I__inference_dense_346_layer_call_and_return_conditional_losses_1128688755

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
f
J__inference_flatten_86_layer_call_and_return_conditional_losses_1128688167

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ4:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ç

V__inference_batch_normalization_86_layer_call_and_return_conditional_losses_1128688820

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Ü
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü	
â
I__inference_dense_346_layer_call_and_return_conditional_losses_1128688240

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü0
Õ
V__inference_batch_normalization_86_layer_call_and_return_conditional_losses_1128688800

inputs
assignmovingavg_1128688775 
assignmovingavg_1_1128688781)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1Ð
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg/1128688775*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1128688775*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOpö
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg/1128688775*
_output_shapes	
:2
AssignMovingAvg/subí
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg/1128688775*
_output_shapes	
:2
AssignMovingAvg/mul·
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1128688775AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg/1128688775*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÖ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*/
_class%
#!loc:@AssignMovingAvg_1/1128688781*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1128688781*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*/
_class%
#!loc:@AssignMovingAvg_1/1128688781*
_output_shapes	
:2
AssignMovingAvg_1/sub÷
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*/
_class%
#!loc:@AssignMovingAvg_1/1128688781*
_output_shapes	
:2
AssignMovingAvg_1/mulÃ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1128688781AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*/
_class%
#!loc:@AssignMovingAvg_1/1128688781*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1´
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§
«
&__inference__traced_restore_1128689133
file_prefix%
!assignvariableop_dense_344_kernel%
!assignvariableop_1_dense_344_bias'
#assignvariableop_2_dense_345_kernel%
!assignvariableop_3_dense_345_bias'
#assignvariableop_4_dense_346_kernel%
!assignvariableop_5_dense_346_bias3
/assignvariableop_6_batch_normalization_86_gamma2
.assignvariableop_7_batch_normalization_86_beta9
5assignvariableop_8_batch_normalization_86_moving_mean=
9assignvariableop_9_batch_normalization_86_moving_variance(
$assignvariableop_10_dense_347_kernel&
"assignvariableop_11_dense_347_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count/
+assignvariableop_19_adam_dense_344_kernel_m-
)assignvariableop_20_adam_dense_344_bias_m/
+assignvariableop_21_adam_dense_345_kernel_m-
)assignvariableop_22_adam_dense_345_bias_m/
+assignvariableop_23_adam_dense_346_kernel_m-
)assignvariableop_24_adam_dense_346_bias_m;
7assignvariableop_25_adam_batch_normalization_86_gamma_m:
6assignvariableop_26_adam_batch_normalization_86_beta_m/
+assignvariableop_27_adam_dense_347_kernel_m-
)assignvariableop_28_adam_dense_347_bias_m/
+assignvariableop_29_adam_dense_344_kernel_v-
)assignvariableop_30_adam_dense_344_bias_v/
+assignvariableop_31_adam_dense_345_kernel_v-
)assignvariableop_32_adam_dense_345_bias_v/
+assignvariableop_33_adam_dense_346_kernel_v-
)assignvariableop_34_adam_dense_346_bias_v;
7assignvariableop_35_adam_batch_normalization_86_gamma_v:
6assignvariableop_36_adam_batch_normalization_86_beta_v/
+assignvariableop_37_adam_dense_347_kernel_v-
)assignvariableop_38_adam_dense_347_bias_v
identity_40¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*¡
valueB(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÞ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesö
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¶
_output_shapes£
 ::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_dense_344_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¦
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_344_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¨
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_345_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_345_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¨
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_346_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¦
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_346_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6´
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_86_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7³
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_86_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8º
AssignVariableOp_8AssignVariableOp5assignvariableop_8_batch_normalization_86_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¾
AssignVariableOp_9AssignVariableOp9assignvariableop_9_batch_normalization_86_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¬
AssignVariableOp_10AssignVariableOp$assignvariableop_10_dense_347_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ª
AssignVariableOp_11AssignVariableOp"assignvariableop_11_dense_347_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12¥
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13§
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14§
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¦
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16®
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¡
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¡
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19³
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_dense_344_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20±
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_344_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21³
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_345_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22±
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_345_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23³
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_346_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24±
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_346_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¿
AssignVariableOp_25AssignVariableOp7assignvariableop_25_adam_batch_normalization_86_gamma_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¾
AssignVariableOp_26AssignVariableOp6assignvariableop_26_adam_batch_normalization_86_beta_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27³
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_347_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28±
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_347_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29³
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_344_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30±
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_344_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31³
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_345_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32±
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_345_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33³
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_dense_346_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34±
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_346_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35¿
AssignVariableOp_35AssignVariableOp7assignvariableop_35_adam_batch_normalization_86_gamma_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36¾
AssignVariableOp_36AssignVariableOp6assignvariableop_36_adam_batch_normalization_86_beta_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37³
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_dense_347_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38±
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_dense_347_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_389
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp¸
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_39«
Identity_40IdentityIdentity_39:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_40"#
identity_40Identity_40:output:0*³
_input_shapes¡
: :::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¤"

H__inference_model_86_layer_call_and_return_conditional_losses_1128688353
input_87
dense_344_1128688323
dense_344_1128688325
dense_345_1128688328
dense_345_1128688330
dense_346_1128688333
dense_346_1128688335%
!batch_normalization_86_1128688338%
!batch_normalization_86_1128688340%
!batch_normalization_86_1128688342%
!batch_normalization_86_1128688344
dense_347_1128688347
dense_347_1128688349
identity¢.batch_normalization_86/StatefulPartitionedCall¢!dense_344/StatefulPartitionedCall¢!dense_345/StatefulPartitionedCall¢!dense_346/StatefulPartitionedCall¢!dense_347/StatefulPartitionedCallá
flatten_86/PartitionedCallPartitionedCallinput_87*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_flatten_86_layer_call_and_return_conditional_losses_11286881672
flatten_86/PartitionedCallÃ
!dense_344/StatefulPartitionedCallStatefulPartitionedCall#flatten_86/PartitionedCall:output:0dense_344_1128688323dense_344_1128688325*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_344_layer_call_and_return_conditional_losses_11286881862#
!dense_344/StatefulPartitionedCallÊ
!dense_345/StatefulPartitionedCallStatefulPartitionedCall*dense_344/StatefulPartitionedCall:output:0dense_345_1128688328dense_345_1128688330*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_345_layer_call_and_return_conditional_losses_11286882132#
!dense_345/StatefulPartitionedCallÊ
!dense_346/StatefulPartitionedCallStatefulPartitionedCall*dense_345/StatefulPartitionedCall:output:0dense_346_1128688333dense_346_1128688335*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_346_layer_call_and_return_conditional_losses_11286882402#
!dense_346/StatefulPartitionedCallÕ
.batch_normalization_86/StatefulPartitionedCallStatefulPartitionedCall*dense_346/StatefulPartitionedCall:output:0!batch_normalization_86_1128688338!batch_normalization_86_1128688340!batch_normalization_86_1128688342!batch_normalization_86_1128688344*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_batch_normalization_86_layer_call_and_return_conditional_losses_112868814620
.batch_normalization_86/StatefulPartitionedCallÖ
!dense_347/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_86/StatefulPartitionedCall:output:0dense_347_1128688347dense_347_1128688349*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_347_layer_call_and_return_conditional_losses_11286883022#
!dense_347/StatefulPartitionedCall¿
IdentityIdentity*dense_347/StatefulPartitionedCall:output:0/^batch_normalization_86/StatefulPartitionedCall"^dense_344/StatefulPartitionedCall"^dense_345/StatefulPartitionedCall"^dense_346/StatefulPartitionedCall"^dense_347/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2`
.batch_normalization_86/StatefulPartitionedCall.batch_normalization_86/StatefulPartitionedCall2F
!dense_344/StatefulPartitionedCall!dense_344/StatefulPartitionedCall2F
!dense_345/StatefulPartitionedCall!dense_345/StatefulPartitionedCall2F
!dense_346/StatefulPartitionedCall!dense_346/StatefulPartitionedCall2F
!dense_347/StatefulPartitionedCall!dense_347/StatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
input_87
þ	
â
I__inference_dense_347_layer_call_and_return_conditional_losses_1128688302

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯	

-__inference_model_86_layer_call_fn_1128688693

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_model_86_layer_call_and_return_conditional_losses_11286884532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ü	
â
I__inference_dense_345_layer_call_and_return_conditional_losses_1128688213

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
G
¿	
H__inference_model_86_layer_call_and_return_conditional_losses_1128688635

inputs,
(dense_344_matmul_readvariableop_resource-
)dense_344_biasadd_readvariableop_resource,
(dense_345_matmul_readvariableop_resource-
)dense_345_biasadd_readvariableop_resource,
(dense_346_matmul_readvariableop_resource-
)dense_346_biasadd_readvariableop_resource<
8batch_normalization_86_batchnorm_readvariableop_resource@
<batch_normalization_86_batchnorm_mul_readvariableop_resource>
:batch_normalization_86_batchnorm_readvariableop_1_resource>
:batch_normalization_86_batchnorm_readvariableop_2_resource,
(dense_347_matmul_readvariableop_resource-
)dense_347_biasadd_readvariableop_resource
identity¢/batch_normalization_86/batchnorm/ReadVariableOp¢1batch_normalization_86/batchnorm/ReadVariableOp_1¢1batch_normalization_86/batchnorm/ReadVariableOp_2¢3batch_normalization_86/batchnorm/mul/ReadVariableOp¢ dense_344/BiasAdd/ReadVariableOp¢dense_344/MatMul/ReadVariableOp¢ dense_345/BiasAdd/ReadVariableOp¢dense_345/MatMul/ReadVariableOp¢ dense_346/BiasAdd/ReadVariableOp¢dense_346/MatMul/ReadVariableOp¢ dense_347/BiasAdd/ReadVariableOp¢dense_347/MatMul/ReadVariableOpu
flatten_86/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_86/Const
flatten_86/ReshapeReshapeinputsflatten_86/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_86/Reshape­
dense_344/MatMul/ReadVariableOpReadVariableOp(dense_344_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_344/MatMul/ReadVariableOp§
dense_344/MatMulMatMulflatten_86/Reshape:output:0'dense_344/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_344/MatMul«
 dense_344/BiasAdd/ReadVariableOpReadVariableOp)dense_344_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_344/BiasAdd/ReadVariableOpª
dense_344/BiasAddBiasAdddense_344/MatMul:product:0(dense_344/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_344/BiasAddw
dense_344/ReluReludense_344/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_344/Relu­
dense_345/MatMul/ReadVariableOpReadVariableOp(dense_345_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_345/MatMul/ReadVariableOp¨
dense_345/MatMulMatMuldense_344/Relu:activations:0'dense_345/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_345/MatMul«
 dense_345/BiasAdd/ReadVariableOpReadVariableOp)dense_345_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_345/BiasAdd/ReadVariableOpª
dense_345/BiasAddBiasAdddense_345/MatMul:product:0(dense_345/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_345/BiasAddw
dense_345/ReluReludense_345/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_345/Relu­
dense_346/MatMul/ReadVariableOpReadVariableOp(dense_346_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_346/MatMul/ReadVariableOp¨
dense_346/MatMulMatMuldense_345/Relu:activations:0'dense_346/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_346/MatMul«
 dense_346/BiasAdd/ReadVariableOpReadVariableOp)dense_346_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_346/BiasAdd/ReadVariableOpª
dense_346/BiasAddBiasAdddense_346/MatMul:product:0(dense_346/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_346/BiasAddw
dense_346/ReluReludense_346/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_346/ReluØ
/batch_normalization_86/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_86_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype021
/batch_normalization_86/batchnorm/ReadVariableOp
&batch_normalization_86/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_86/batchnorm/add/yå
$batch_normalization_86/batchnorm/addAddV27batch_normalization_86/batchnorm/ReadVariableOp:value:0/batch_normalization_86/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2&
$batch_normalization_86/batchnorm/add©
&batch_normalization_86/batchnorm/RsqrtRsqrt(batch_normalization_86/batchnorm/add:z:0*
T0*
_output_shapes	
:2(
&batch_normalization_86/batchnorm/Rsqrtä
3batch_normalization_86/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_86_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype025
3batch_normalization_86/batchnorm/mul/ReadVariableOpâ
$batch_normalization_86/batchnorm/mulMul*batch_normalization_86/batchnorm/Rsqrt:y:0;batch_normalization_86/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2&
$batch_normalization_86/batchnorm/mulÒ
&batch_normalization_86/batchnorm/mul_1Muldense_346/Relu:activations:0(batch_normalization_86/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_86/batchnorm/mul_1Þ
1batch_normalization_86/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_86_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype023
1batch_normalization_86/batchnorm/ReadVariableOp_1â
&batch_normalization_86/batchnorm/mul_2Mul9batch_normalization_86/batchnorm/ReadVariableOp_1:value:0(batch_normalization_86/batchnorm/mul:z:0*
T0*
_output_shapes	
:2(
&batch_normalization_86/batchnorm/mul_2Þ
1batch_normalization_86/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_86_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype023
1batch_normalization_86/batchnorm/ReadVariableOp_2à
$batch_normalization_86/batchnorm/subSub9batch_normalization_86/batchnorm/ReadVariableOp_2:value:0*batch_normalization_86/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2&
$batch_normalization_86/batchnorm/subâ
&batch_normalization_86/batchnorm/add_1AddV2*batch_normalization_86/batchnorm/mul_1:z:0(batch_normalization_86/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_86/batchnorm/add_1¬
dense_347/MatMul/ReadVariableOpReadVariableOp(dense_347_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02!
dense_347/MatMul/ReadVariableOpµ
dense_347/MatMulMatMul*batch_normalization_86/batchnorm/add_1:z:0'dense_347/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_347/MatMulª
 dense_347/BiasAdd/ReadVariableOpReadVariableOp)dense_347_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_347/BiasAdd/ReadVariableOp©
dense_347/BiasAddBiasAdddense_347/MatMul:product:0(dense_347/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_347/BiasAdd
dense_347/SoftmaxSoftmaxdense_347/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_347/SoftmaxÓ
IdentityIdentitydense_347/Softmax:softmax:00^batch_normalization_86/batchnorm/ReadVariableOp2^batch_normalization_86/batchnorm/ReadVariableOp_12^batch_normalization_86/batchnorm/ReadVariableOp_24^batch_normalization_86/batchnorm/mul/ReadVariableOp!^dense_344/BiasAdd/ReadVariableOp ^dense_344/MatMul/ReadVariableOp!^dense_345/BiasAdd/ReadVariableOp ^dense_345/MatMul/ReadVariableOp!^dense_346/BiasAdd/ReadVariableOp ^dense_346/MatMul/ReadVariableOp!^dense_347/BiasAdd/ReadVariableOp ^dense_347/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2b
/batch_normalization_86/batchnorm/ReadVariableOp/batch_normalization_86/batchnorm/ReadVariableOp2f
1batch_normalization_86/batchnorm/ReadVariableOp_11batch_normalization_86/batchnorm/ReadVariableOp_12f
1batch_normalization_86/batchnorm/ReadVariableOp_21batch_normalization_86/batchnorm/ReadVariableOp_22j
3batch_normalization_86/batchnorm/mul/ReadVariableOp3batch_normalization_86/batchnorm/mul/ReadVariableOp2D
 dense_344/BiasAdd/ReadVariableOp dense_344/BiasAdd/ReadVariableOp2B
dense_344/MatMul/ReadVariableOpdense_344/MatMul/ReadVariableOp2D
 dense_345/BiasAdd/ReadVariableOp dense_345/BiasAdd/ReadVariableOp2B
dense_345/MatMul/ReadVariableOpdense_345/MatMul/ReadVariableOp2D
 dense_346/BiasAdd/ReadVariableOp dense_346/BiasAdd/ReadVariableOp2B
dense_346/MatMul/ReadVariableOpdense_346/MatMul/ReadVariableOp2D
 dense_347/BiasAdd/ReadVariableOp dense_347/BiasAdd/ReadVariableOp2B
dense_347/MatMul/ReadVariableOpdense_347/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ç

V__inference_batch_normalization_86_layer_call_and_return_conditional_losses_1128688146

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Ü
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦S
ö

%__inference__wrapped_model_1128688017
input_875
1model_86_dense_344_matmul_readvariableop_resource6
2model_86_dense_344_biasadd_readvariableop_resource5
1model_86_dense_345_matmul_readvariableop_resource6
2model_86_dense_345_biasadd_readvariableop_resource5
1model_86_dense_346_matmul_readvariableop_resource6
2model_86_dense_346_biasadd_readvariableop_resourceE
Amodel_86_batch_normalization_86_batchnorm_readvariableop_resourceI
Emodel_86_batch_normalization_86_batchnorm_mul_readvariableop_resourceG
Cmodel_86_batch_normalization_86_batchnorm_readvariableop_1_resourceG
Cmodel_86_batch_normalization_86_batchnorm_readvariableop_2_resource5
1model_86_dense_347_matmul_readvariableop_resource6
2model_86_dense_347_biasadd_readvariableop_resource
identity¢8model_86/batch_normalization_86/batchnorm/ReadVariableOp¢:model_86/batch_normalization_86/batchnorm/ReadVariableOp_1¢:model_86/batch_normalization_86/batchnorm/ReadVariableOp_2¢<model_86/batch_normalization_86/batchnorm/mul/ReadVariableOp¢)model_86/dense_344/BiasAdd/ReadVariableOp¢(model_86/dense_344/MatMul/ReadVariableOp¢)model_86/dense_345/BiasAdd/ReadVariableOp¢(model_86/dense_345/MatMul/ReadVariableOp¢)model_86/dense_346/BiasAdd/ReadVariableOp¢(model_86/dense_346/MatMul/ReadVariableOp¢)model_86/dense_347/BiasAdd/ReadVariableOp¢(model_86/dense_347/MatMul/ReadVariableOp
model_86/flatten_86/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
model_86/flatten_86/Const¦
model_86/flatten_86/ReshapeReshapeinput_87"model_86/flatten_86/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_86/flatten_86/ReshapeÈ
(model_86/dense_344/MatMul/ReadVariableOpReadVariableOp1model_86_dense_344_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02*
(model_86/dense_344/MatMul/ReadVariableOpË
model_86/dense_344/MatMulMatMul$model_86/flatten_86/Reshape:output:00model_86/dense_344/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_86/dense_344/MatMulÆ
)model_86/dense_344/BiasAdd/ReadVariableOpReadVariableOp2model_86_dense_344_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model_86/dense_344/BiasAdd/ReadVariableOpÎ
model_86/dense_344/BiasAddBiasAdd#model_86/dense_344/MatMul:product:01model_86/dense_344/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_86/dense_344/BiasAdd
model_86/dense_344/ReluRelu#model_86/dense_344/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_86/dense_344/ReluÈ
(model_86/dense_345/MatMul/ReadVariableOpReadVariableOp1model_86_dense_345_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02*
(model_86/dense_345/MatMul/ReadVariableOpÌ
model_86/dense_345/MatMulMatMul%model_86/dense_344/Relu:activations:00model_86/dense_345/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_86/dense_345/MatMulÆ
)model_86/dense_345/BiasAdd/ReadVariableOpReadVariableOp2model_86_dense_345_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model_86/dense_345/BiasAdd/ReadVariableOpÎ
model_86/dense_345/BiasAddBiasAdd#model_86/dense_345/MatMul:product:01model_86/dense_345/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_86/dense_345/BiasAdd
model_86/dense_345/ReluRelu#model_86/dense_345/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_86/dense_345/ReluÈ
(model_86/dense_346/MatMul/ReadVariableOpReadVariableOp1model_86_dense_346_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02*
(model_86/dense_346/MatMul/ReadVariableOpÌ
model_86/dense_346/MatMulMatMul%model_86/dense_345/Relu:activations:00model_86/dense_346/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_86/dense_346/MatMulÆ
)model_86/dense_346/BiasAdd/ReadVariableOpReadVariableOp2model_86_dense_346_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model_86/dense_346/BiasAdd/ReadVariableOpÎ
model_86/dense_346/BiasAddBiasAdd#model_86/dense_346/MatMul:product:01model_86/dense_346/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_86/dense_346/BiasAdd
model_86/dense_346/ReluRelu#model_86/dense_346/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_86/dense_346/Reluó
8model_86/batch_normalization_86/batchnorm/ReadVariableOpReadVariableOpAmodel_86_batch_normalization_86_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02:
8model_86/batch_normalization_86/batchnorm/ReadVariableOp§
/model_86/batch_normalization_86/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:21
/model_86/batch_normalization_86/batchnorm/add/y
-model_86/batch_normalization_86/batchnorm/addAddV2@model_86/batch_normalization_86/batchnorm/ReadVariableOp:value:08model_86/batch_normalization_86/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2/
-model_86/batch_normalization_86/batchnorm/addÄ
/model_86/batch_normalization_86/batchnorm/RsqrtRsqrt1model_86/batch_normalization_86/batchnorm/add:z:0*
T0*
_output_shapes	
:21
/model_86/batch_normalization_86/batchnorm/Rsqrtÿ
<model_86/batch_normalization_86/batchnorm/mul/ReadVariableOpReadVariableOpEmodel_86_batch_normalization_86_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02>
<model_86/batch_normalization_86/batchnorm/mul/ReadVariableOp
-model_86/batch_normalization_86/batchnorm/mulMul3model_86/batch_normalization_86/batchnorm/Rsqrt:y:0Dmodel_86/batch_normalization_86/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2/
-model_86/batch_normalization_86/batchnorm/mulö
/model_86/batch_normalization_86/batchnorm/mul_1Mul%model_86/dense_346/Relu:activations:01model_86/batch_normalization_86/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/model_86/batch_normalization_86/batchnorm/mul_1ù
:model_86/batch_normalization_86/batchnorm/ReadVariableOp_1ReadVariableOpCmodel_86_batch_normalization_86_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02<
:model_86/batch_normalization_86/batchnorm/ReadVariableOp_1
/model_86/batch_normalization_86/batchnorm/mul_2MulBmodel_86/batch_normalization_86/batchnorm/ReadVariableOp_1:value:01model_86/batch_normalization_86/batchnorm/mul:z:0*
T0*
_output_shapes	
:21
/model_86/batch_normalization_86/batchnorm/mul_2ù
:model_86/batch_normalization_86/batchnorm/ReadVariableOp_2ReadVariableOpCmodel_86_batch_normalization_86_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02<
:model_86/batch_normalization_86/batchnorm/ReadVariableOp_2
-model_86/batch_normalization_86/batchnorm/subSubBmodel_86/batch_normalization_86/batchnorm/ReadVariableOp_2:value:03model_86/batch_normalization_86/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2/
-model_86/batch_normalization_86/batchnorm/sub
/model_86/batch_normalization_86/batchnorm/add_1AddV23model_86/batch_normalization_86/batchnorm/mul_1:z:01model_86/batch_normalization_86/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/model_86/batch_normalization_86/batchnorm/add_1Ç
(model_86/dense_347/MatMul/ReadVariableOpReadVariableOp1model_86_dense_347_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(model_86/dense_347/MatMul/ReadVariableOpÙ
model_86/dense_347/MatMulMatMul3model_86/batch_normalization_86/batchnorm/add_1:z:00model_86/dense_347/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_86/dense_347/MatMulÅ
)model_86/dense_347/BiasAdd/ReadVariableOpReadVariableOp2model_86_dense_347_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model_86/dense_347/BiasAdd/ReadVariableOpÍ
model_86/dense_347/BiasAddBiasAdd#model_86/dense_347/MatMul:product:01model_86/dense_347/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_86/dense_347/BiasAdd
model_86/dense_347/SoftmaxSoftmax#model_86/dense_347/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_86/dense_347/SoftmaxÈ
IdentityIdentity$model_86/dense_347/Softmax:softmax:09^model_86/batch_normalization_86/batchnorm/ReadVariableOp;^model_86/batch_normalization_86/batchnorm/ReadVariableOp_1;^model_86/batch_normalization_86/batchnorm/ReadVariableOp_2=^model_86/batch_normalization_86/batchnorm/mul/ReadVariableOp*^model_86/dense_344/BiasAdd/ReadVariableOp)^model_86/dense_344/MatMul/ReadVariableOp*^model_86/dense_345/BiasAdd/ReadVariableOp)^model_86/dense_345/MatMul/ReadVariableOp*^model_86/dense_346/BiasAdd/ReadVariableOp)^model_86/dense_346/MatMul/ReadVariableOp*^model_86/dense_347/BiasAdd/ReadVariableOp)^model_86/dense_347/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2t
8model_86/batch_normalization_86/batchnorm/ReadVariableOp8model_86/batch_normalization_86/batchnorm/ReadVariableOp2x
:model_86/batch_normalization_86/batchnorm/ReadVariableOp_1:model_86/batch_normalization_86/batchnorm/ReadVariableOp_12x
:model_86/batch_normalization_86/batchnorm/ReadVariableOp_2:model_86/batch_normalization_86/batchnorm/ReadVariableOp_22|
<model_86/batch_normalization_86/batchnorm/mul/ReadVariableOp<model_86/batch_normalization_86/batchnorm/mul/ReadVariableOp2V
)model_86/dense_344/BiasAdd/ReadVariableOp)model_86/dense_344/BiasAdd/ReadVariableOp2T
(model_86/dense_344/MatMul/ReadVariableOp(model_86/dense_344/MatMul/ReadVariableOp2V
)model_86/dense_345/BiasAdd/ReadVariableOp)model_86/dense_345/BiasAdd/ReadVariableOp2T
(model_86/dense_345/MatMul/ReadVariableOp(model_86/dense_345/MatMul/ReadVariableOp2V
)model_86/dense_346/BiasAdd/ReadVariableOp)model_86/dense_346/BiasAdd/ReadVariableOp2T
(model_86/dense_346/MatMul/ReadVariableOp(model_86/dense_346/MatMul/ReadVariableOp2V
)model_86/dense_347/BiasAdd/ReadVariableOp)model_86/dense_347/BiasAdd/ReadVariableOp2T
(model_86/dense_347/MatMul/ReadVariableOp(model_86/dense_347/MatMul/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
input_87
¨
K
/__inference_flatten_86_layer_call_fn_1128688704

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_flatten_86_layer_call_and_return_conditional_losses_11286881672
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ4:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ü	
â
I__inference_dense_344_layer_call_and_return_conditional_losses_1128688186

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­	

-__inference_model_86_layer_call_fn_1128688664

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_model_86_layer_call_and_return_conditional_losses_11286883902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ío
µ

H__inference_model_86_layer_call_and_return_conditional_losses_1128688585

inputs,
(dense_344_matmul_readvariableop_resource-
)dense_344_biasadd_readvariableop_resource,
(dense_345_matmul_readvariableop_resource-
)dense_345_biasadd_readvariableop_resource,
(dense_346_matmul_readvariableop_resource-
)dense_346_biasadd_readvariableop_resource5
1batch_normalization_86_assignmovingavg_11286885537
3batch_normalization_86_assignmovingavg_1_1128688559@
<batch_normalization_86_batchnorm_mul_readvariableop_resource<
8batch_normalization_86_batchnorm_readvariableop_resource,
(dense_347_matmul_readvariableop_resource-
)dense_347_biasadd_readvariableop_resource
identity¢:batch_normalization_86/AssignMovingAvg/AssignSubVariableOp¢5batch_normalization_86/AssignMovingAvg/ReadVariableOp¢<batch_normalization_86/AssignMovingAvg_1/AssignSubVariableOp¢7batch_normalization_86/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_86/batchnorm/ReadVariableOp¢3batch_normalization_86/batchnorm/mul/ReadVariableOp¢ dense_344/BiasAdd/ReadVariableOp¢dense_344/MatMul/ReadVariableOp¢ dense_345/BiasAdd/ReadVariableOp¢dense_345/MatMul/ReadVariableOp¢ dense_346/BiasAdd/ReadVariableOp¢dense_346/MatMul/ReadVariableOp¢ dense_347/BiasAdd/ReadVariableOp¢dense_347/MatMul/ReadVariableOpu
flatten_86/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_86/Const
flatten_86/ReshapeReshapeinputsflatten_86/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_86/Reshape­
dense_344/MatMul/ReadVariableOpReadVariableOp(dense_344_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_344/MatMul/ReadVariableOp§
dense_344/MatMulMatMulflatten_86/Reshape:output:0'dense_344/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_344/MatMul«
 dense_344/BiasAdd/ReadVariableOpReadVariableOp)dense_344_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_344/BiasAdd/ReadVariableOpª
dense_344/BiasAddBiasAdddense_344/MatMul:product:0(dense_344/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_344/BiasAddw
dense_344/ReluReludense_344/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_344/Relu­
dense_345/MatMul/ReadVariableOpReadVariableOp(dense_345_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_345/MatMul/ReadVariableOp¨
dense_345/MatMulMatMuldense_344/Relu:activations:0'dense_345/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_345/MatMul«
 dense_345/BiasAdd/ReadVariableOpReadVariableOp)dense_345_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_345/BiasAdd/ReadVariableOpª
dense_345/BiasAddBiasAdddense_345/MatMul:product:0(dense_345/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_345/BiasAddw
dense_345/ReluReludense_345/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_345/Relu­
dense_346/MatMul/ReadVariableOpReadVariableOp(dense_346_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_346/MatMul/ReadVariableOp¨
dense_346/MatMulMatMuldense_345/Relu:activations:0'dense_346/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_346/MatMul«
 dense_346/BiasAdd/ReadVariableOpReadVariableOp)dense_346_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_346/BiasAdd/ReadVariableOpª
dense_346/BiasAddBiasAdddense_346/MatMul:product:0(dense_346/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_346/BiasAddw
dense_346/ReluReludense_346/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_346/Relu¸
5batch_normalization_86/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_86/moments/mean/reduction_indicesë
#batch_normalization_86/moments/meanMeandense_346/Relu:activations:0>batch_normalization_86/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2%
#batch_normalization_86/moments/meanÂ
+batch_normalization_86/moments/StopGradientStopGradient,batch_normalization_86/moments/mean:output:0*
T0*
_output_shapes
:	2-
+batch_normalization_86/moments/StopGradient
0batch_normalization_86/moments/SquaredDifferenceSquaredDifferencedense_346/Relu:activations:04batch_normalization_86/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_86/moments/SquaredDifferenceÀ
9batch_normalization_86/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_86/moments/variance/reduction_indices
'batch_normalization_86/moments/varianceMean4batch_normalization_86/moments/SquaredDifference:z:0Bbatch_normalization_86/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2)
'batch_normalization_86/moments/varianceÆ
&batch_normalization_86/moments/SqueezeSqueeze,batch_normalization_86/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2(
&batch_normalization_86/moments/SqueezeÎ
(batch_normalization_86/moments/Squeeze_1Squeeze0batch_normalization_86/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2*
(batch_normalization_86/moments/Squeeze_1
,batch_normalization_86/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_86/AssignMovingAvg/1128688553*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_86/AssignMovingAvg/decayÝ
5batch_normalization_86/AssignMovingAvg/ReadVariableOpReadVariableOp1batch_normalization_86_assignmovingavg_1128688553*
_output_shapes	
:*
dtype027
5batch_normalization_86/AssignMovingAvg/ReadVariableOpé
*batch_normalization_86/AssignMovingAvg/subSub=batch_normalization_86/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_86/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_86/AssignMovingAvg/1128688553*
_output_shapes	
:2,
*batch_normalization_86/AssignMovingAvg/subà
*batch_normalization_86/AssignMovingAvg/mulMul.batch_normalization_86/AssignMovingAvg/sub:z:05batch_normalization_86/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_86/AssignMovingAvg/1128688553*
_output_shapes	
:2,
*batch_normalization_86/AssignMovingAvg/mulÁ
:batch_normalization_86/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp1batch_normalization_86_assignmovingavg_1128688553.batch_normalization_86/AssignMovingAvg/mul:z:06^batch_normalization_86/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_86/AssignMovingAvg/1128688553*
_output_shapes
 *
dtype02<
:batch_normalization_86/AssignMovingAvg/AssignSubVariableOp
.batch_normalization_86/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*F
_class<
:8loc:@batch_normalization_86/AssignMovingAvg_1/1128688559*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_86/AssignMovingAvg_1/decayã
7batch_normalization_86/AssignMovingAvg_1/ReadVariableOpReadVariableOp3batch_normalization_86_assignmovingavg_1_1128688559*
_output_shapes	
:*
dtype029
7batch_normalization_86/AssignMovingAvg_1/ReadVariableOpó
,batch_normalization_86/AssignMovingAvg_1/subSub?batch_normalization_86/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_86/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*F
_class<
:8loc:@batch_normalization_86/AssignMovingAvg_1/1128688559*
_output_shapes	
:2.
,batch_normalization_86/AssignMovingAvg_1/subê
,batch_normalization_86/AssignMovingAvg_1/mulMul0batch_normalization_86/AssignMovingAvg_1/sub:z:07batch_normalization_86/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*F
_class<
:8loc:@batch_normalization_86/AssignMovingAvg_1/1128688559*
_output_shapes	
:2.
,batch_normalization_86/AssignMovingAvg_1/mulÍ
<batch_normalization_86/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp3batch_normalization_86_assignmovingavg_1_11286885590batch_normalization_86/AssignMovingAvg_1/mul:z:08^batch_normalization_86/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*F
_class<
:8loc:@batch_normalization_86/AssignMovingAvg_1/1128688559*
_output_shapes
 *
dtype02>
<batch_normalization_86/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_86/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_86/batchnorm/add/yß
$batch_normalization_86/batchnorm/addAddV21batch_normalization_86/moments/Squeeze_1:output:0/batch_normalization_86/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2&
$batch_normalization_86/batchnorm/add©
&batch_normalization_86/batchnorm/RsqrtRsqrt(batch_normalization_86/batchnorm/add:z:0*
T0*
_output_shapes	
:2(
&batch_normalization_86/batchnorm/Rsqrtä
3batch_normalization_86/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_86_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype025
3batch_normalization_86/batchnorm/mul/ReadVariableOpâ
$batch_normalization_86/batchnorm/mulMul*batch_normalization_86/batchnorm/Rsqrt:y:0;batch_normalization_86/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2&
$batch_normalization_86/batchnorm/mulÒ
&batch_normalization_86/batchnorm/mul_1Muldense_346/Relu:activations:0(batch_normalization_86/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_86/batchnorm/mul_1Ø
&batch_normalization_86/batchnorm/mul_2Mul/batch_normalization_86/moments/Squeeze:output:0(batch_normalization_86/batchnorm/mul:z:0*
T0*
_output_shapes	
:2(
&batch_normalization_86/batchnorm/mul_2Ø
/batch_normalization_86/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_86_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype021
/batch_normalization_86/batchnorm/ReadVariableOpÞ
$batch_normalization_86/batchnorm/subSub7batch_normalization_86/batchnorm/ReadVariableOp:value:0*batch_normalization_86/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2&
$batch_normalization_86/batchnorm/subâ
&batch_normalization_86/batchnorm/add_1AddV2*batch_normalization_86/batchnorm/mul_1:z:0(batch_normalization_86/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_86/batchnorm/add_1¬
dense_347/MatMul/ReadVariableOpReadVariableOp(dense_347_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02!
dense_347/MatMul/ReadVariableOpµ
dense_347/MatMulMatMul*batch_normalization_86/batchnorm/add_1:z:0'dense_347/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_347/MatMulª
 dense_347/BiasAdd/ReadVariableOpReadVariableOp)dense_347_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_347/BiasAdd/ReadVariableOp©
dense_347/BiasAddBiasAdddense_347/MatMul:product:0(dense_347/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_347/BiasAdd
dense_347/SoftmaxSoftmaxdense_347/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_347/SoftmaxÙ
IdentityIdentitydense_347/Softmax:softmax:0;^batch_normalization_86/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_86/AssignMovingAvg/ReadVariableOp=^batch_normalization_86/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_86/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_86/batchnorm/ReadVariableOp4^batch_normalization_86/batchnorm/mul/ReadVariableOp!^dense_344/BiasAdd/ReadVariableOp ^dense_344/MatMul/ReadVariableOp!^dense_345/BiasAdd/ReadVariableOp ^dense_345/MatMul/ReadVariableOp!^dense_346/BiasAdd/ReadVariableOp ^dense_346/MatMul/ReadVariableOp!^dense_347/BiasAdd/ReadVariableOp ^dense_347/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2x
:batch_normalization_86/AssignMovingAvg/AssignSubVariableOp:batch_normalization_86/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_86/AssignMovingAvg/ReadVariableOp5batch_normalization_86/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_86/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_86/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_86/AssignMovingAvg_1/ReadVariableOp7batch_normalization_86/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_86/batchnorm/ReadVariableOp/batch_normalization_86/batchnorm/ReadVariableOp2j
3batch_normalization_86/batchnorm/mul/ReadVariableOp3batch_normalization_86/batchnorm/mul/ReadVariableOp2D
 dense_344/BiasAdd/ReadVariableOp dense_344/BiasAdd/ReadVariableOp2B
dense_344/MatMul/ReadVariableOpdense_344/MatMul/ReadVariableOp2D
 dense_345/BiasAdd/ReadVariableOp dense_345/BiasAdd/ReadVariableOp2B
dense_345/MatMul/ReadVariableOpdense_345/MatMul/ReadVariableOp2D
 dense_346/BiasAdd/ReadVariableOp dense_346/BiasAdd/ReadVariableOp2B
dense_346/MatMul/ReadVariableOpdense_346/MatMul/ReadVariableOp2D
 dense_347/BiasAdd/ReadVariableOp dense_347/BiasAdd/ReadVariableOp2B
dense_347/MatMul/ReadVariableOpdense_347/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
³	

-__inference_model_86_layer_call_fn_1128688417
input_87
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinput_87unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_model_86_layer_call_and_return_conditional_losses_11286883902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
input_87
Å
®
;__inference_batch_normalization_86_layer_call_fn_1128688846

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_batch_normalization_86_layer_call_and_return_conditional_losses_11286881462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü	
â
I__inference_dense_345_layer_call_and_return_conditional_losses_1128688735

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢"

H__inference_model_86_layer_call_and_return_conditional_losses_1128688319
input_87
dense_344_1128688197
dense_344_1128688199
dense_345_1128688224
dense_345_1128688226
dense_346_1128688251
dense_346_1128688253%
!batch_normalization_86_1128688282%
!batch_normalization_86_1128688284%
!batch_normalization_86_1128688286%
!batch_normalization_86_1128688288
dense_347_1128688313
dense_347_1128688315
identity¢.batch_normalization_86/StatefulPartitionedCall¢!dense_344/StatefulPartitionedCall¢!dense_345/StatefulPartitionedCall¢!dense_346/StatefulPartitionedCall¢!dense_347/StatefulPartitionedCallá
flatten_86/PartitionedCallPartitionedCallinput_87*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_flatten_86_layer_call_and_return_conditional_losses_11286881672
flatten_86/PartitionedCallÃ
!dense_344/StatefulPartitionedCallStatefulPartitionedCall#flatten_86/PartitionedCall:output:0dense_344_1128688197dense_344_1128688199*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_344_layer_call_and_return_conditional_losses_11286881862#
!dense_344/StatefulPartitionedCallÊ
!dense_345/StatefulPartitionedCallStatefulPartitionedCall*dense_344/StatefulPartitionedCall:output:0dense_345_1128688224dense_345_1128688226*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_345_layer_call_and_return_conditional_losses_11286882132#
!dense_345/StatefulPartitionedCallÊ
!dense_346/StatefulPartitionedCallStatefulPartitionedCall*dense_345/StatefulPartitionedCall:output:0dense_346_1128688251dense_346_1128688253*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_346_layer_call_and_return_conditional_losses_11286882402#
!dense_346/StatefulPartitionedCallÓ
.batch_normalization_86/StatefulPartitionedCallStatefulPartitionedCall*dense_346/StatefulPartitionedCall:output:0!batch_normalization_86_1128688282!batch_normalization_86_1128688284!batch_normalization_86_1128688286!batch_normalization_86_1128688288*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_batch_normalization_86_layer_call_and_return_conditional_losses_112868811320
.batch_normalization_86/StatefulPartitionedCallÖ
!dense_347/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_86/StatefulPartitionedCall:output:0dense_347_1128688313dense_347_1128688315*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_347_layer_call_and_return_conditional_losses_11286883022#
!dense_347/StatefulPartitionedCall¿
IdentityIdentity*dense_347/StatefulPartitionedCall:output:0/^batch_normalization_86/StatefulPartitionedCall"^dense_344/StatefulPartitionedCall"^dense_345/StatefulPartitionedCall"^dense_346/StatefulPartitionedCall"^dense_347/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2`
.batch_normalization_86/StatefulPartitionedCall.batch_normalization_86/StatefulPartitionedCall2F
!dense_344/StatefulPartitionedCall!dense_344/StatefulPartitionedCall2F
!dense_345/StatefulPartitionedCall!dense_345/StatefulPartitionedCall2F
!dense_346/StatefulPartitionedCall!dense_346/StatefulPartitionedCall2F
!dense_347/StatefulPartitionedCall!dense_347/StatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
input_87
ë

.__inference_dense_344_layer_call_fn_1128688724

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_344_layer_call_and_return_conditional_losses_11286881862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*²
serving_default
A
input_875
serving_default_input_87:0ÿÿÿÿÿÿÿÿÿ4=
	dense_3470
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ïã
û=
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	optimizer
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
*t&call_and_return_all_conditional_losses
u_default_save_signature
v__call__"Ó:
_tf_keras_network·:{"class_name": "Functional", "name": "model_86", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_86", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 52, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_87"}, "name": "input_87", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_86", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_86", "inbound_nodes": [[["input_87", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_344", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_344", "inbound_nodes": [[["flatten_86", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_345", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_345", "inbound_nodes": [[["dense_344", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_346", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_346", "inbound_nodes": [[["dense_345", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_86", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_86", "inbound_nodes": [[["dense_346", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_347", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_347", "inbound_nodes": [[["batch_normalization_86", 0, 0, {}]]]}], "input_layers": [["input_87", 0, 0]], "output_layers": [["dense_347", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 52, 3]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 52, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_86", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 52, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_87"}, "name": "input_87", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_86", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_86", "inbound_nodes": [[["input_87", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_344", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_344", "inbound_nodes": [[["flatten_86", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_345", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_345", "inbound_nodes": [[["dense_344", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_346", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_346", "inbound_nodes": [[["dense_345", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_86", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_86", "inbound_nodes": [[["dense_346", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_347", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_347", "inbound_nodes": [[["batch_normalization_86", 0, 0, {}]]]}], "input_layers": [["input_87", 0, 0]], "output_layers": [["dense_347", 0, 0]]}}, "training_config": {"loss": "loss_func", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "clipnorm": 1, "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ó"ð
_tf_keras_input_layerÐ{"class_name": "InputLayer", "name": "input_87", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 52, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 52, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_87"}}
è
regularization_losses
	variables
trainable_variables
	keras_api
*w&call_and_return_all_conditional_losses
x__call__"Ù
_tf_keras_layer¿{"class_name": "Flatten", "name": "flatten_86", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_86", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*y&call_and_return_all_conditional_losses
z__call__"ó
_tf_keras_layerÙ{"class_name": "Dense", "name": "dense_344", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_344", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 156}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 156]}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*{&call_and_return_all_conditional_losses
|__call__"ó
_tf_keras_layerÙ{"class_name": "Dense", "name": "dense_345", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_345", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}


kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
*}&call_and_return_all_conditional_losses
~__call__"ó
_tf_keras_layerÙ{"class_name": "Dense", "name": "dense_346", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_346", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
·	
$axis
	%gamma
&beta
'moving_mean
(moving_variance
)regularization_losses
*	variables
+trainable_variables
,	keras_api
*&call_and_return_all_conditional_losses
__call__"â
_tf_keras_layerÈ{"class_name": "BatchNormalization", "name": "batch_normalization_86", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_86", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
ú

-kernel
.bias
/regularization_losses
0	variables
1trainable_variables
2	keras_api
+&call_and_return_all_conditional_losses
__call__"Ó
_tf_keras_layer¹{"class_name": "Dense", "name": "dense_347", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_347", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}

3iter

4beta_1

5beta_2
	6decay
7learning_ratem`mambmcmdme%mf&mg-mh.mivjvkvlvmvnvo%vp&vq-vr.vs"
	optimizer
 "
trackable_list_wrapper
v
0
1
2
3
4
5
%6
&7
'8
(9
-10
.11"
trackable_list_wrapper
f
0
1
2
3
4
5
%6
&7
-8
.9"
trackable_list_wrapper
Ê

8layers
9metrics
	regularization_losses
:non_trainable_variables

	variables
;layer_regularization_losses
<layer_metrics
trainable_variables
v__call__
u_default_save_signature
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
=metrics

>layers
regularization_losses
?non_trainable_variables
	variables
@layer_regularization_losses
Alayer_metrics
trainable_variables
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
$:"
2dense_344/kernel
:2dense_344/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Bmetrics

Clayers
regularization_losses
Dnon_trainable_variables
	variables
Elayer_regularization_losses
Flayer_metrics
trainable_variables
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
$:"
2dense_345/kernel
:2dense_345/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Gmetrics

Hlayers
regularization_losses
Inon_trainable_variables
	variables
Jlayer_regularization_losses
Klayer_metrics
trainable_variables
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
$:"
2dense_346/kernel
:2dense_346/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Lmetrics

Mlayers
 regularization_losses
Nnon_trainable_variables
!	variables
Olayer_regularization_losses
Player_metrics
"trainable_variables
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2batch_normalization_86/gamma
*:(2batch_normalization_86/beta
3:1 (2"batch_normalization_86/moving_mean
7:5 (2&batch_normalization_86/moving_variance
 "
trackable_list_wrapper
<
%0
&1
'2
(3"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
®
Qmetrics

Rlayers
)regularization_losses
Snon_trainable_variables
*	variables
Tlayer_regularization_losses
Ulayer_metrics
+trainable_variables
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
#:!	2dense_347/kernel
:2dense_347/bias
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
°
Vmetrics

Wlayers
/regularization_losses
Xnon_trainable_variables
0	variables
Ylayer_regularization_losses
Zlayer_metrics
1trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
'
[0"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
»
	\total
	]count
^	variables
_	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
\0
]1"
trackable_list_wrapper
-
^	variables"
_generic_user_object
):'
2Adam/dense_344/kernel/m
": 2Adam/dense_344/bias/m
):'
2Adam/dense_345/kernel/m
": 2Adam/dense_345/bias/m
):'
2Adam/dense_346/kernel/m
": 2Adam/dense_346/bias/m
0:.2#Adam/batch_normalization_86/gamma/m
/:-2"Adam/batch_normalization_86/beta/m
(:&	2Adam/dense_347/kernel/m
!:2Adam/dense_347/bias/m
):'
2Adam/dense_344/kernel/v
": 2Adam/dense_344/bias/v
):'
2Adam/dense_345/kernel/v
": 2Adam/dense_345/bias/v
):'
2Adam/dense_346/kernel/v
": 2Adam/dense_346/bias/v
0:.2#Adam/batch_normalization_86/gamma/v
/:-2"Adam/batch_normalization_86/beta/v
(:&	2Adam/dense_347/kernel/v
!:2Adam/dense_347/bias/v
î2ë
H__inference_model_86_layer_call_and_return_conditional_losses_1128688319
H__inference_model_86_layer_call_and_return_conditional_losses_1128688585
H__inference_model_86_layer_call_and_return_conditional_losses_1128688353
H__inference_model_86_layer_call_and_return_conditional_losses_1128688635À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
è2å
%__inference__wrapped_model_1128688017»
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *+¢(
&#
input_87ÿÿÿÿÿÿÿÿÿ4
2ÿ
-__inference_model_86_layer_call_fn_1128688664
-__inference_model_86_layer_call_fn_1128688417
-__inference_model_86_layer_call_fn_1128688480
-__inference_model_86_layer_call_fn_1128688693À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ô2ñ
J__inference_flatten_86_layer_call_and_return_conditional_losses_1128688699¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ù2Ö
/__inference_flatten_86_layer_call_fn_1128688704¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_dense_344_layer_call_and_return_conditional_losses_1128688715¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
.__inference_dense_344_layer_call_fn_1128688724¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_dense_345_layer_call_and_return_conditional_losses_1128688735¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
.__inference_dense_345_layer_call_fn_1128688744¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_dense_346_layer_call_and_return_conditional_losses_1128688755¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
.__inference_dense_346_layer_call_fn_1128688764¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
V__inference_batch_normalization_86_layer_call_and_return_conditional_losses_1128688820
V__inference_batch_normalization_86_layer_call_and_return_conditional_losses_1128688800´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
´2±
;__inference_batch_normalization_86_layer_call_fn_1128688833
;__inference_batch_normalization_86_layer_call_fn_1128688846´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ó2ð
I__inference_dense_347_layer_call_and_return_conditional_losses_1128688857¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
.__inference_dense_347_layer_call_fn_1128688866¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÐBÍ
(__inference_signature_wrapper_1128688519input_87"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ¥
%__inference__wrapped_model_1128688017|(%'&-.5¢2
+¢(
&#
input_87ÿÿÿÿÿÿÿÿÿ4
ª "5ª2
0
	dense_347# 
	dense_347ÿÿÿÿÿÿÿÿÿ¾
V__inference_batch_normalization_86_layer_call_and_return_conditional_losses_1128688800d'(%&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¾
V__inference_batch_normalization_86_layer_call_and_return_conditional_losses_1128688820d(%'&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
;__inference_batch_normalization_86_layer_call_fn_1128688833W'(%&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
;__inference_batch_normalization_86_layer_call_fn_1128688846W(%'&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ«
I__inference_dense_344_layer_call_and_return_conditional_losses_1128688715^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_dense_344_layer_call_fn_1128688724Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ«
I__inference_dense_345_layer_call_and_return_conditional_losses_1128688735^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_dense_345_layer_call_fn_1128688744Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ«
I__inference_dense_346_layer_call_and_return_conditional_losses_1128688755^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_dense_346_layer_call_fn_1128688764Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
I__inference_dense_347_layer_call_and_return_conditional_losses_1128688857]-.0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_dense_347_layer_call_fn_1128688866P-.0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ«
J__inference_flatten_86_layer_call_and_return_conditional_losses_1128688699]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ4
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_flatten_86_layer_call_fn_1128688704P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ4
ª "ÿÿÿÿÿÿÿÿÿÀ
H__inference_model_86_layer_call_and_return_conditional_losses_1128688319t'(%&-.=¢:
3¢0
&#
input_87ÿÿÿÿÿÿÿÿÿ4
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 À
H__inference_model_86_layer_call_and_return_conditional_losses_1128688353t(%'&-.=¢:
3¢0
&#
input_87ÿÿÿÿÿÿÿÿÿ4
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
H__inference_model_86_layer_call_and_return_conditional_losses_1128688585r'(%&-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
H__inference_model_86_layer_call_and_return_conditional_losses_1128688635r(%'&-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_model_86_layer_call_fn_1128688417g'(%&-.=¢:
3¢0
&#
input_87ÿÿÿÿÿÿÿÿÿ4
p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_model_86_layer_call_fn_1128688480g(%'&-.=¢:
3¢0
&#
input_87ÿÿÿÿÿÿÿÿÿ4
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_model_86_layer_call_fn_1128688664e'(%&-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_model_86_layer_call_fn_1128688693e(%'&-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p 

 
ª "ÿÿÿÿÿÿÿÿÿµ
(__inference_signature_wrapper_1128688519(%'&-.A¢>
¢ 
7ª4
2
input_87&#
input_87ÿÿÿÿÿÿÿÿÿ4"5ª2
0
	dense_347# 
	dense_347ÿÿÿÿÿÿÿÿÿ