; ModuleID = 'test.cpp'
source_filename = "test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z6matmulPA400_fPA300_fS2_(ptr noundef %A, ptr noundef %B, ptr noundef %C) #0 {
entry:
  %A.addr = alloca ptr, align 8
  %B.addr = alloca ptr, align 8
  %C.addr = alloca ptr, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  %i1 = alloca i32, align 4
  %j2 = alloca i32, align 4
  %k6 = alloca i32, align 4
  store ptr %A, ptr %A.addr, align 8
  store ptr %B, ptr %B.addr, align 8
  store ptr %C, ptr %C.addr, align 8
  store i32 0, ptr %i1, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc23, %entry
  %0 = load i32, ptr %i1, align 4
  %cmp = icmp slt i32 %0, 200
  br i1 %cmp, label %for.body, label %for.end25

for.body:                                         ; preds = %for.cond
  store i32 0, ptr %j2, align 4
  br label %for.cond3

for.cond3:                                        ; preds = %for.inc20, %for.body
  %1 = load i32, ptr %j2, align 4
  %cmp4 = icmp slt i32 %1, 300
  br i1 %cmp4, label %for.body5, label %for.end22

for.body5:                                        ; preds = %for.cond3
  store i32 0, ptr %k6, align 4
  br label %for.cond7

for.cond7:                                        ; preds = %for.inc, %for.body5
  %2 = load i32, ptr %k6, align 4
  %cmp8 = icmp slt i32 %2, 400
  br i1 %cmp8, label %for.body9, label %for.end

for.body9:                                        ; preds = %for.cond7
  %3 = load ptr, ptr %A.addr, align 8
  %4 = load i32, ptr %i1, align 4
  %idxprom = sext i32 %4 to i64
  %arrayidx = getelementptr inbounds [400 x float], ptr %3, i64 %idxprom
  %5 = load i32, ptr %k6, align 4
  %idxprom10 = sext i32 %5 to i64
  %arrayidx11 = getelementptr inbounds [400 x float], ptr %arrayidx, i64 0, i64 %idxprom10
  %6 = load float, ptr %arrayidx11, align 4
  %7 = load ptr, ptr %B.addr, align 8
  %8 = load i32, ptr %k6, align 4
  %idxprom12 = sext i32 %8 to i64
  %arrayidx13 = getelementptr inbounds [300 x float], ptr %7, i64 %idxprom12
  %9 = load i32, ptr %j2, align 4
  %idxprom14 = sext i32 %9 to i64
  %arrayidx15 = getelementptr inbounds [300 x float], ptr %arrayidx13, i64 0, i64 %idxprom14
  %10 = load float, ptr %arrayidx15, align 4
  %11 = load ptr, ptr %C.addr, align 8
  %12 = load i32, ptr %i1, align 4
  %idxprom16 = sext i32 %12 to i64
  %arrayidx17 = getelementptr inbounds [300 x float], ptr %11, i64 %idxprom16
  %13 = load i32, ptr %j2, align 4
  %idxprom18 = sext i32 %13 to i64
  %arrayidx19 = getelementptr inbounds [300 x float], ptr %arrayidx17, i64 0, i64 %idxprom18
  %14 = load float, ptr %arrayidx19, align 4
  %15 = call float @llvm.fmuladd.f32(float %6, float %10, float %14)
  store float %15, ptr %arrayidx19, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body9
  %16 = load i32, ptr %k6, align 4
  %inc = add nsw i32 %16, 1
  store i32 %inc, ptr %k6, align 4
  br label %for.cond7, !llvm.loop !6

for.end:                                          ; preds = %for.cond7
  br label %for.inc20

for.inc20:                                        ; preds = %for.end
  %17 = load i32, ptr %j2, align 4
  %inc21 = add nsw i32 %17, 1
  store i32 %inc21, ptr %j2, align 4
  br label %for.cond3, !llvm.loop !8

for.end22:                                        ; preds = %for.cond3
  br label %for.inc23

for.inc23:                                        ; preds = %for.end22
  %18 = load i32, ptr %i1, align 4
  %inc24 = add nsw i32 %18, 1
  store i32 %inc24, ptr %i1, align 4
  br label %for.cond, !llvm.loop !9

for.end25:                                        ; preds = %for.cond
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #1

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 22.0.0git (https://github.com/llvm/llvm-project.git 0a2eb850d0dbd9caa1b22ee94f3b2b9903f679cb)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
!9 = distinct !{!9, !7}
