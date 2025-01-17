package com.example.next;

import android.annotation.SuppressLint;
import android.content.Context;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import android.os.Handler;
import android.os.Message;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import com.google.gson.Gson;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * A simple {@link Fragment} subclass.
 * Use the {@link Fragment1#newInstance} factory method to
 * create an instance of this fragment.
 */
public class Fragment1 extends Fragment {

    // TODO: Rename parameter arguments, choose names that match
    // the fragment initialization parameters, e.g. ARG_ITEM_NUMBER
    private static final String ARG_PARAM1 = "param1";
    private static final String ARG_PARAM2 = "param2";
    private BirthdaybookAdapter mBirthdaybookAdapter;

    // TODO: Rename and change types of parameters
    private String mParam1;
    private String mParam2;
    private Button mButton;
    private String mUrl2="http://apis.juhe.cn/fapig/birthdayBook/query?key=961a80eb755f89d429566afb8610aa4b&keyword=2025-01-09";
    private MyHandler mHandler;

    public Fragment1() {
        // Required empty public constructor
    }

    /**
     * Use this factory method to create a new instance of
     * this fragment using the provided parameters.
     *
     * @param param1 Parameter 1.
     * @param param2 Parameter 2.
     * @return A new instance of fragment Fragment1.
     */
    // TODO: Rename and change types and number of parameters
    public static Fragment1 newInstance(String param1, String param2) {
        Fragment1 fragment = new Fragment1();
        Bundle args = new Bundle();
        args.putString(ARG_PARAM1, param1);
        args.putString(ARG_PARAM2, param2);
        fragment.setArguments(args);
        return fragment;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (getArguments() != null) {
            mParam1 = getArguments().getString(ARG_PARAM1);
            mParam2 = getArguments().getString(ARG_PARAM2);
        }
    }

    public class MyHandler extends Handler
    {

        @Override
        public void handleMessage(@NonNull Message msg) {
            super.handleMessage(msg);
            String data = (String) msg.obj;
            BirthdaybookData birthdaybookData= new Gson().fromJson(data, BirthdaybookData.class);
            birthdaybookData.data=new ArrayList<>();
            if (msg.what == 100)
            {
                //刷新适配器
                if (null != mBirthdaybookAdapter)
                {
                    birthdaybookData.addData(birthdaybookData.getResult());//data不在处理数据的类型里面，所以data需要实例化后赋值
                    mBirthdaybookAdapter.setListData(birthdaybookData.getData());
                }
            }
        }
    }

    @Override
    public View onCreateView(LayoutInflater inflater, @NonNull ViewGroup container,
                             @NonNull Bundle savedIns )
    {
        View view =inflater.inflate(R.layout.fragment_1, container, false);
        LinearLayoutManager linearLayoutManager=new LinearLayoutManager(getActivity());
        RecyclerView mRecyclerView1 = view.findViewById(R.id.recyclerView1);;
        LinearLayoutManager layoutManager = new LinearLayoutManager(getActivity());
        mRecyclerView1.setLayoutManager(layoutManager);
        mButton=view.findViewById(R.id.Button1);
        mBirthdaybookAdapter=new BirthdaybookAdapter(Fragment1.this);
        mRecyclerView1.setAdapter(mBirthdaybookAdapter);
        HashMap<String,String> params=new HashMap<>();
        params.put("2025-01-09","123456");
        mHandler = new MyHandler();
        JsonDatapost jsonData1= new JsonDatapost(mHandler,params);
        mButton.setOnClickListener(v -> {jsonData1.sendPostRequest(mUrl2);});
        return view;
    }
}